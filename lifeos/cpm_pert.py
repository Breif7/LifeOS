"""
CPM / PERT Engine & Monte Carlo Risk Simulation
Implements PDM Graph, Forward & Backward passes, Slack, Critical Path,
and stochastic PERT Monte Carlo analysis.
"""

from copy import deepcopy
from typing import Dict, List, Tuple
import numpy as np

from lifeos.config import WBS_CATALOG, WorkPackage


class CPMNetwork:
    """Critical Path Method (CPM) and PERT analysis network."""

    def __init__(self, packages: Dict[str, WorkPackage] = None):
        if packages is None:
            self.packages = deepcopy(WBS_CATALOG)
        else:
            self.packages = deepcopy(packages)
        self.critical_path: List[str] = []
        self.project_duration: float = 0.0

    def compute_schedule(self) -> Dict[str, WorkPackage]:
        """Performs Forward and Backward passes to calculate ES, EF, LS, LF, Slack, and Criticality."""
        # Build adjacency graph
        successors: Dict[str, List[str]] = {wbs_id: [] for wbs_id in self.packages}
        for wbs_id, pkg in self.packages.items():
            for p in pkg.predecessors:
                if p in successors:
                    successors[p].append(wbs_id)

        # Topological sorting (Kahn's algorithm)
        in_degree = {wbs_id: len(pkg.predecessors) for wbs_id, pkg in self.packages.items()}
        queue = [wbs_id for wbs_id, deg in in_degree.items() if deg == 0]
        topo_order = []

        while queue:
            curr = queue.pop(0)
            topo_order.append(curr)
            for succ in successors[curr]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        if len(topo_order) != len(self.packages):
            raise ValueError("Cycle detected in WBS precedence graph!")

        # 1. Forward Pass (Early Start / Early Finish)
        for wbs_id in topo_order:
            pkg = self.packages[wbs_id]
            if not pkg.predecessors:
                pkg.es = 0.0
            else:
                pkg.es = max(self.packages[p].ef for p in pkg.predecessors)
            pkg.ef = pkg.es + pkg.duration

        self.project_duration = max(pkg.ef for pkg in self.packages.values())

        # 2. Backward Pass (Late Start / Late Finish)
        for wbs_id in reversed(topo_order):
            pkg = self.packages[wbs_id]
            succs = successors[wbs_id]
            if not succs:
                pkg.lf = self.project_duration
            else:
                pkg.lf = min(self.packages[s].ls for s in succs)
            pkg.ls = pkg.lf - pkg.duration
            pkg.total_slack = round(pkg.ls - pkg.es, 2)
            pkg.is_critical = abs(pkg.total_slack) < 1e-4

        # Identify critical path sequence
        self.critical_path = [
            wbs_id for wbs_id in topo_order if self.packages[wbs_id].is_critical
        ]

        return self.packages

    def get_summary_table(self) -> List[Dict]:
        """Returns structured dictionary rows for tabulating or DataFrame creation."""
        if not self.critical_path:
            self.compute_schedule()

        rows = []
        for wbs_id, pkg in self.packages.items():
            rows.append({
                "wbs_id": pkg.wbs_id,
                "name": pkg.name,
                "sprint": pkg.sprint,
                "duration": pkg.duration,
                "bac_cost": pkg.bac_cost,
                "predecessors": ", ".join(pkg.predecessors) if pkg.predecessors else "-",
                "es": pkg.es,
                "ef": pkg.ef,
                "ls": pkg.ls,
                "lf": pkg.lf,
                "slack": pkg.total_slack,
                "pert_te": pkg.pert_te,
                "is_critical": "SÍ" if pkg.is_critical else "No",
            })
        return rows

    def get_pert_statistics(self) -> Dict[str, float]:
        """Calculates expected PERT duration and variance along the critical path."""
        if not self.critical_path:
            self.compute_schedule()

        pert_mean_cp = sum(self.packages[cid].pert_te for cid in self.critical_path)
        pert_var_cp = sum(self.packages[cid].pert_variance for cid in self.critical_path)
        pert_std_cp = np.sqrt(pert_var_cp)

        return {
            "deterministic_cp_hours": sum(self.packages[cid].duration for cid in self.critical_path),
            "total_scope_hours": sum(pkg.duration for pkg in self.packages.values()),
            "pert_mean_cp_hours": round(float(pert_mean_cp), 2),
            "pert_std_cp_hours": round(float(pert_std_cp), 2),
            "pert_var_cp": round(float(pert_var_cp), 4),
        }

    def run_monte_carlo_simulation(
        self,
        iterations: int = 10000,
        target_hours: float = 255.0,
        random_seed: int = 42,
    ) -> Dict[str, any]:
        """
        Executes Monte Carlo simulation (10,000 iterations) sampling from Beta-PERT distribution.
        Returns simulated durations array, percentiles, and probability of completing within target_hours.
        """
        np.random.seed(random_seed)
        simulated_durations = np.zeros(iterations)

        # Precompute topo order and graph structure
        self.compute_schedule()
        successors: Dict[str, List[str]] = {wbs_id: [] for wbs_id in self.packages}
        for wbs_id, pkg in self.packages.items():
            for p in pkg.predecessors:
                if p in successors:
                    successors[p].append(wbs_id)

        in_degree = {wbs_id: len(pkg.predecessors) for wbs_id, pkg in self.packages.items()}
        queue = [wbs_id for wbs_id, deg in in_degree.items() if deg == 0]
        topo_order = []
        while queue:
            curr = queue.pop(0)
            topo_order.append(curr)
            for succ in successors[curr]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        # Sample durations for each task using Beta-PERT parameters:
        # alpha = 1 + 4*(M - O)/(P - O), beta = 1 + 4*(P - M)/(P - O)
        samples = {}
        for wbs_id, pkg in self.packages.items():
            o, m, p = pkg.pert_o, pkg.pert_m, pkg.pert_p
            if p > o:
                alpha = 1.0 + 4.0 * (m - o) / (p - o)
                beta_param = 1.0 + 4.0 * (p - m) / (p - o)
                # Beta distribution in [0, 1] scaled to [o, p]
                beta_samples = np.random.beta(alpha, beta_param, size=iterations)
                samples[wbs_id] = o + beta_samples * (p - o)
            else:
                samples[wbs_id] = np.full(iterations, m)

        # Compute critical project duration for each simulation iter
        for it in range(iterations):
            ef_map = {}
            for wbs_id in topo_order:
                pkg = self.packages[wbs_id]
                dur = samples[wbs_id][it]
                if not pkg.predecessors:
                    es = 0.0
                else:
                    es = max(ef_map[p] for p in pkg.predecessors)
                ef_map[wbs_id] = es + dur
            simulated_durations[it] = max(ef_map.values())

        prob_success = float(np.mean(simulated_durations <= target_hours) * 100.0)
        p50 = float(np.percentile(simulated_durations, 50))
        p85 = float(np.percentile(simulated_durations, 85))
        p90 = float(np.percentile(simulated_durations, 90))
        p95 = float(np.percentile(simulated_durations, 95))

        return {
            "iterations": iterations,
            "mean_hours": round(float(np.mean(simulated_durations)), 2),
            "std_hours": round(float(np.std(simulated_durations)), 2),
            "min_hours": round(float(np.min(simulated_durations)), 2),
            "max_hours": round(float(np.max(simulated_durations)), 2),
            "p50_hours": round(p50, 2),
            "p85_hours": round(p85, 2),
            "p90_hours": round(p90, 2),
            "p95_hours": round(p95, 2),
            "prob_under_threshold": round(prob_success, 1),
            "threshold_hours": target_hours,
            "raw_simulations": simulated_durations,
        }
