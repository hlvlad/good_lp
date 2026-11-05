//! A solver that uses [Gurobi](https://www.gurobi.com/) through the native
//! `grb` bindings (a *linked library*, not the `gurobi_cl` command line).
//!
//! This mirrors the [`cplex`](super::cplex) backend: it translates a good_lp
//! [`UnsolvedProblem`] into a `grb::Model`, solves it in-process, and exposes
//! the variable values through the [`Solution`] trait.
//!
//! # Environment
//! The `grb` crate links `libgurobiXXX.so` at build time. Point it at your
//! Gurobi install with `GUROBI_HOME` (and, when the shared object is not the
//! default name for the enabled feature flag, `GUROBI_LIBNAME`, e.g.
//! `GUROBI_LIBNAME=gurobi130`). A license valid for the *C API* is required at
//! run time (a `TYPE=PIP` license only authorises the `gurobipy` module and is
//! rejected here).

use std::collections::HashMap;

use crate::{
    constraint::ConstraintReference,
    solvers::{
        InPlaceSolution, ModelWithMutableObjective, ModelWithMutableRhs, ObjectiveDirection,
        ReusableModel, WithTimeLimit,
    },
    variable::UnsolvedProblem,
    Constraint, Expression, ResolutionError, Solution, SolutionStatus, SolverModel, Variable,
    VariableDefinition,
};

use grb::constr::IneqExpr;
use grb::prelude::*;

thread_local! {
    /// A Gurobi environment that writes no log file.
    ///
    /// An empty log filename tells Gurobi not to create one. Held per thread
    /// and reused, mirroring how `grb` holds its default: acquiring an
    /// environment validates the licence, which is far too costly to repeat for
    /// every one of the dozens of solves an epsilon-constraint sweep issues.
    /// Log file is bound here, at environment creation -- Gurobi opens it
    /// before any model parameter can be set, which is why `OutputFlag` on the
    /// model cannot turn logging back on. `GOOD_LP_GUROBI_LOG` names a path to
    /// log to; unset (the default) keeps the environment silent.
    static QUIET_ENV: Env = Env::new(
        &std::env::var("GOOD_LP_GUROBI_LOG").unwrap_or_default()
    ).expect("Unable to create Gurobi environment");
}

/// The [Gurobi](https://www.gurobi.com/) solver, to be used with
/// [`UnsolvedProblem::using`].
///
/// ```ignore
/// use good_lp::{variables, variable, gurobi, SolverModel, Solution};
/// let mut vars = variables!();
/// let x = vars.add(variable().binary());
/// let sol = vars.maximise(x).using(gurobi).solve().unwrap();
/// assert_eq!(sol.value(x), 1.0);
/// ```
pub fn gurobi(to_solve: UnsolvedProblem) -> GurobiProblem {
    // Built on a private environment rather than `Model::new`'s default one,
    // which `grb` hardcodes as `Env::new("gurobi.log")` -- so every process
    // using this backend opens a `gurobi.log` in its working directory and
    // appends to it for as long as it runs. `OutputFlag` does not help: the
    // file is opened when the *environment* is created, before any parameter
    // can be set on the model. Left alone it grows without bound; two of them
    // had reached 156MB between them.
    let mut model =
        QUIET_ENV.with(|env| Model::with_env("good_lp", env).expect("Unable to create Gurobi model"));

    // Keep the log off by default: augmecon runs dozens of solves per instance
    // and the Gurobi banner/log otherwise floods stdout. When a log file was
    // requested the environment above is already writing to it, so leave output
    // enabled rather than silencing what the caller asked for.
    if std::env::var_os("GOOD_LP_GUROBI_LOG").is_none() {
        let _ = model.set_param(param::OutputFlag, 0);
    }

    let sense = match to_solve.direction {
        ObjectiveDirection::Maximisation => ModelSense::Maximize,
        ObjectiveDirection::Minimisation => ModelSense::Minimize,
    };

    let mut id_for_var = HashMap::new();
    for (
        var,
        &VariableDefinition {
            min,
            max,
            is_integer,
            ref name,
            ..
        },
    ) in to_solve.variables.iter_variables_with_def()
    {
        let vtype = if is_integer {
            if min == 0. && max == 1. {
                VarType::Binary
            } else {
                VarType::Integer
            }
        } else {
            VarType::Continuous
        };
        // good_lp uses ±inf for unbounded; Gurobi's sentinel is ±GRB_INFINITY.
        let lb = if min.is_infinite() {
            -grb::INFINITY
        } else {
            min
        };
        let ub = if max.is_infinite() {
            grb::INFINITY
        } else {
            max
        };
        // Objective coefficients are applied below via `set_objective`, so add
        // the variable with a zero objective coefficient here.
        let g_var = model
            .add_var(name, vtype, 0.0, lb, ub, std::iter::empty())
            .expect("Unable to add Gurobi variable");
        id_for_var.insert(var, g_var);
    }

    // Objective: Σ cⱼ·xⱼ + constant. The constant does not change the argmin/argmax
    // but is included so Gurobi's ObjVal matches the good_lp objective faithfully.
    let mut objective = grb::expr::LinExpr::new();
    for (var, &coeff) in &to_solve.objective.linear.coefficients {
        objective.add_term(coeff, id_for_var[var]);
    }
    objective.add_constant(to_solve.objective.constant);
    model
        .set_objective(objective, sense)
        .expect("Unable to set Gurobi objective");

    GurobiProblem {
        model,
        id_for_var,
        constrs: Vec::new(),
    }
}

/// A Gurobi model, ready to have constraints added and to be solved.
pub struct GurobiProblem {
    model: grb::Model,
    id_for_var: HashMap<Variable, grb::Var>,
    /// Native constraint handles, indexed by [`ConstraintReference::index`]
    /// (constraints are only appended, so the index is stable). Enables
    /// in-place right-hand-side updates via [`ModelWithMutableRhs`].
    constrs: Vec<grb::Constr>,
}

impl GurobiProblem {
    /// The `good_lp` variable to `grb::Var` mapping built when the model was
    /// created.
    ///
    /// Needed to reach Gurobi features that are set per variable and have no
    /// `good_lp` equivalent -- notably the `ObjN` attribute used to declare a
    /// hierarchical (lexicographic) multi-objective model.
    pub fn var_map(&self) -> &HashMap<Variable, grb::Var> {
        &self.id_for_var
    }

    /// Access the underlying `grb::Model`.
    pub fn as_inner(&self) -> &grb::Model {
        &self.model
    }

    /// Mutably access the underlying `grb::Model` (to set extra parameters,
    /// e.g. `set_param(param::Threads, 1)`).
    pub fn as_inner_mut(&mut self) -> &mut grb::Model {
        &mut self.model
    }
}

impl WithTimeLimit for GurobiProblem {
    fn with_time_limit<T: Into<f64>>(mut self, seconds: T) -> Self {
        self.model
            .set_param(param::TimeLimit, seconds.into())
            .expect("Unable to set Gurobi time limit");
        self
    }
}

impl SolverModel for GurobiProblem {
    type Solution = GurobiSolved;
    type Error = ResolutionError;

    fn solve(mut self) -> Result<Self::Solution, Self::Error> {
        let status = self.optimize_and_classify()?;
        Ok(GurobiSolved::new(self, status))
    }

    /// Add every constraint in one Gurobi call.
    ///
    /// The default implementation loops over [`SolverModel::add_constraint`],
    /// which costs a `GRBaddconstr` *and* a `GRBgetintparam` per row -- `grb`
    /// re-reads `UpdateMode` after each add to decide whether the new object is
    /// immediately visible. On a 27905-row model that measured 640ms, against
    /// 35ms to create the model and all of its variables: 95% of the cost of
    /// building a model was the row loop, and it is paid on every solve.
    ///
    /// Requires the vendored `grb`; the released crate's `add_constrs`
    /// segfaults and misnumbers the handles it returns.
    fn with_all(mut self, constraints: impl IntoIterator<Item = Constraint>) -> Self {
        let named: Vec<(String, IneqExpr)> = constraints
            .into_iter()
            .map(|c| {
                // good_lp constraints are `expr <sense> 0`, i.e. Sum cj.xj + k
                // {<=,=} 0, rewritten as Sum cj.xj {<=,=} -k. Same rewrite as
                // the single-constraint path.
                let rhs = -c.expression.constant;
                let mut lhs = grb::expr::LinExpr::new();
                for (var, &coeff) in &c.expression.linear.coefficients {
                    lhs.add_term(coeff, self.id_for_var[var]);
                }
                let sense = if c.is_equality {
                    ConstrSense::Equal
                } else {
                    ConstrSense::Less
                };
                // Unnamed constraints pass "" -- Gurobi auto-names them
                // (R0, R1, ...), avoiding a String allocation per row.
                (
                    c.name.unwrap_or_default(),
                    IneqExpr {
                        lhs: Expr::from(lhs),
                        sense,
                        rhs: Expr::Constant(rhs),
                    },
                )
            })
            .collect();

        let added = self
            .model
            .add_constrs(named.iter().map(|(name, expr)| (name, expr.clone())))
            .expect("Unable to add constraints to Gurobi model");
        // In supplied order, so `ConstraintReference` indices keep matching the
        // rows they name and `ModelWithMutableRhs` can address them later.
        self.constrs.extend(added);
        self
    }

    fn add_constraint(&mut self, c: Constraint) -> ConstraintReference {
        // good_lp constraints are `expr <sense> 0`, i.e. Σ cⱼ·xⱼ + k {≤,=} 0,
        // which we rewrite as Σ cⱼ·xⱼ {≤,=} -k.
        let rhs = -c.expression.constant;
        let mut lhs = grb::expr::LinExpr::new();
        for (var, &coeff) in &c.expression.linear.coefficients {
            lhs.add_term(coeff, self.id_for_var[var]);
        }
        let sense = if c.is_equality {
            ConstrSense::Equal
        } else {
            ConstrSense::Less
        };
        let index = self.constrs.len();
        // Unnamed constraints pass "" — Gurobi auto-names them (R0, R1, …), which
        // avoids a per-constraint String allocation on large models.
        let name = c.name.as_deref().unwrap_or("");
        let constr = self
            .model
            .add_constr(
                name,
                IneqExpr {
                    lhs: Expr::from(lhs),
                    sense,
                    rhs: Expr::Constant(rhs),
                },
            )
            .expect("Unable to add constraint to Gurobi model");
        // Record the native handle so ModelWithMutableRhs can address it later.
        self.constrs.push(constr);

        ConstraintReference { index }
    }

    fn name() -> &'static str {
        "Gurobi"
    }
}

/// A solved Gurobi problem. Holds the model so variable values can be queried.
pub struct GurobiSolved {
    model: grb::Model,
    id_for_var: HashMap<Variable, grb::Var>,
    status: SolutionStatus,
}

impl GurobiSolved {
    fn new(problem: GurobiProblem, status: SolutionStatus) -> Self {
        Self {
            model: problem.model,
            id_for_var: problem.id_for_var,
            status,
        }
    }

    /// The solved `grb::Model`.
    ///
    /// Exposes what `good_lp`'s own `Solution` cannot: notably the `Status`
    /// attribute, which is the only way to tell a proven optimum from an
    /// incumbent returned when a time limit expired.
    pub fn as_inner(&self) -> &grb::Model {
        &self.model
    }

    /// Every solution Gurobi retained in its solution pool for this solve.
    ///
    /// A MIP solve visits many feasible integer solutions on the way to its
    /// optimum and normally discards all but the best. With `PoolSearchMode`
    /// set they are kept instead, and for a scalarised multi-objective solve
    /// they are valuable: successive incumbents improve the scalarised
    /// objective while ranging freely over the others, so many are
    /// non-dominated in the original objective space. Harvesting them costs no
    /// extra solve.
    ///
    /// Returns one map of variable values per pooled solution, the incumbent
    /// first. Empty when the model is not a MIP or no solution was found.
    pub fn solution_pool(&mut self) -> Vec<HashMap<Variable, f64>> {
        use grb::prelude::*;
        let count = self.model.get_attr(attr::SolCount).unwrap_or(0);
        let mut out = Vec::with_capacity(count.max(0) as usize);
        for k in 0..count {
            // `Xn` reads the solution selected by the SolutionNumber parameter.
            if self.model.set_param(param::SolutionNumber, k).is_err() {
                break;
            }
            let mut values = HashMap::with_capacity(self.id_for_var.len());
            let mut ok = true;
            for (var, g_var) in &self.id_for_var {
                match self.model.get_obj_attr(attr::Xn, g_var) {
                    Ok(v) => {
                        values.insert(*var, v);
                    }
                    Err(_) => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                out.push(values);
            }
        }
        out
    }
}

impl Solution for GurobiSolved {
    fn status(&self) -> SolutionStatus {
        self.status
    }

    fn value(&self, var: Variable) -> f64 {
        let g_var = &self.id_for_var[&var];
        self.model
            .get_obj_attr(attr::X, g_var)
            .expect("Unable to read Gurobi variable value")
    }
}

impl GurobiProblem {
    /// Run `optimize()` and map Gurobi's status to a good_lp [`SolutionStatus`],
    /// or a [`ResolutionError`] when there is no usable solution. Shared by the
    /// consuming [`SolverModel::solve`] and the in-place [`ReusableModel::solve_mut`]
    /// so the two paths never diverge in how they classify a result.
    fn optimize_and_classify(&mut self) -> Result<SolutionStatus, ResolutionError> {
        self.model
            .optimize()
            .map_err(|e| ResolutionError::Str(format!("Gurobi optimize failed: {e:?}")))?;

        let status = self
            .model
            .status()
            .map_err(|e| ResolutionError::Str(format!("Gurobi status query failed: {e:?}")))?;

        // Number of feasible solutions currently stored.
        let sol_count: i32 = self.model.get_attr(attr::SolCount).unwrap_or(0);

        match status {
            Status::Optimal => Ok(SolutionStatus::Optimal),
            // A time/limit stop with at least one incumbent is a usable, non-optimal
            // solution — surface it rather than erroring, matching how the HiGHS/SCIP
            // budget-limited path is used by augmecon's exact phase.
            Status::TimeLimit
            | Status::SubOptimal
            | Status::SolutionLimit
            | Status::NodeLimit
            | Status::IterationLimit
            | Status::Interrupted
                if sol_count > 0 =>
            {
                Ok(SolutionStatus::TimeLimit)
            }
            Status::Infeasible => Err(ResolutionError::Infeasible),
            Status::Unbounded | Status::InfOrUnbd => Err(ResolutionError::Unbounded),
            other => Err(ResolutionError::Str(format!(
                "Gurobi finished with status {other:?} and no feasible solution"
            ))),
        }
    }
}

impl GurobiProblem {
    /// Every solution Gurobi kept for the most recent [`ReusableModel::solve_mut`].
    ///
    /// The same pool [`GurobiSolved::solution_pool`] exposes, reachable from a
    /// model that was solved in place. `solve_mut` hands back an owned snapshot
    /// of the incumbent alone, so without this a reused model would silently
    /// lose the other solutions a solve already found and paid for -- which for
    /// a multiobjective sweep are free nondominated candidates.
    ///
    /// Reads the pool in place: it moves `SolutionNumber` and leaves it at the
    /// last index read, so call it before the next solve, which resets it.
    pub fn solution_pool(&mut self) -> Vec<HashMap<Variable, f64>> {
        use grb::prelude::*;
        let count = self.model.get_attr(attr::SolCount).unwrap_or(0);
        let mut out = Vec::with_capacity(count.max(0) as usize);
        for k in 0..count {
            if self.model.set_param(param::SolutionNumber, k).is_err() {
                break;
            }
            let mut values = HashMap::with_capacity(self.id_for_var.len());
            let mut ok = true;
            for (var, g_var) in &self.id_for_var {
                match self.model.get_obj_attr(attr::Xn, g_var) {
                    Ok(v) => {
                        values.insert(*var, v);
                    }
                    Err(_) => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                out.push(values);
            }
        }
        out
    }
}

impl ReusableModel for GurobiProblem {
    fn solve_mut(&mut self) -> Result<InPlaceSolution, Self::Error> {
        let status = self.optimize_and_classify()?;
        // Snapshot every variable's value into an owned map. This is done once,
        // immediately after optimize(), so the values are consistent and the
        // returned solution never aliases the (still-mutable) model.
        let mut values = HashMap::with_capacity(self.id_for_var.len());
        for (&var, g_var) in &self.id_for_var {
            let v = self
                .model
                .get_obj_attr(attr::X, g_var)
                .map_err(|e| ResolutionError::Str(format!("Gurobi value read failed: {e:?}")))?;
            values.insert(var, v);
        }
        Ok(InPlaceSolution::new(values, status))
    }
}

impl ModelWithMutableRhs for GurobiProblem {
    fn set_rhs(&mut self, constraint: ConstraintReference, rhs: f64) {
        // Flush any pending structural changes so the constraint is committed and
        // addressable even if set_rhs is called before the first solve. This is a
        // no-op once the model is stable (as it is between solves in a sweep), and
        // mirrors what grb's own set_objective does internally.
        self.model
            .update()
            .expect("Unable to update Gurobi model before setting RHS");
        let constr = &self.constrs[constraint.index];
        self.model
            .set_obj_attr(attr::RHS, constr, rhs)
            .expect("Unable to set Gurobi constraint RHS");
    }
}

impl ModelWithMutableObjective for GurobiProblem {
    fn set_objective(&mut self, objective: Expression, sense: ObjectiveDirection) {
        let sense = match sense {
            ObjectiveDirection::Maximisation => ModelSense::Maximize,
            ObjectiveDirection::Minimisation => ModelSense::Minimize,
        };
        let mut expr = grb::expr::LinExpr::new();
        for (var, &coeff) in &objective.linear.coefficients {
            expr.add_term(coeff, self.id_for_var[var]);
        }
        expr.add_constant(objective.constant);
        // grb::Model::set_objective rewrites the objective coefficient of *every*
        // variable (0.0 for any absent from `expr`), so no term from a previous
        // objective can linger — repeated calls are safe.
        self.model
            .set_objective(expr, sense)
            .expect("Unable to set Gurobi objective");
    }
}
