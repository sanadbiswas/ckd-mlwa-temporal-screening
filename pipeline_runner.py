from pathlib import Path
import time


class CKDPipelineRunner:
    """Execute the modular CKD analysis stages in one shared namespace."""

    def __init__(self, project_root):
        self.project_root = Path(project_root).resolve()
        self.namespace = {
            "__name__": "__main__",
            "PROJECT_ROOT": self.project_root,
        }

    def _run(self, relative_path):
        path = self.project_root / relative_path
        if not path.exists():
            raise FileNotFoundError(f"Module not found: {path}")
        code = compile(path.read_text(encoding="utf-8"), str(path), "exec")
        exec(code, self.namespace, self.namespace)
        return self.namespace

    def setup(self):
        return self._run("modules/00_setup.py")

    def prepare_data(self):
        return self._run("modules/01_data_preparation.py")

    def load_development_data(self):
        return self._run("modules/02_development_data.py")

    def build_preprocessor(self):
        return self._run("modules/03_preprocessing.py")

    def define_candidate_models(self):
        return self._run("modules/04_candidate_models.py")

    def define_metrics(self):
        return self._run("modules/05_metrics.py")

    def run_nested_cv(self):
        self._run("modules/06_nested_cv.py")
        return self.namespace.get("nested_summary", self.namespace.get("nested_cv_summary"))

    def run_bootstrap_uncertainty(self):
        return self._run("modules/07_bootstrap_uncertainty.py")

    def run_paired_model_comparison(self):
        return self._run("modules/08_paired_model_comparison.py")

    def tune_final_model(self):
        return self._run("modules/09_final_tuning.py")

    def calibrate_model(self):
        return self._run("modules/10_calibration.py")

    def freeze_threshold(self):
        return self._run("modules/11_screening_threshold.py")

    def fit_final_model(self):
        return self._run("modules/12_final_model.py")

    def build_temporal_cohort(self):
        return self._run("modules/13_temporal_data.py")

    def run_temporal_validation(self):
        return self._run("modules/14_temporal_validation.py")

    def run_temporal_calibration(self):
        return self._run("modules/15_temporal_calibration.py")

    def run_decision_curve(self):
        return self._run("modules/16_decision_curve.py")

    def run_age_sensitivity(self):
        return self._run("modules/17_age_coupling_sensitivity.py")

    def run_outcome_component_sensitivity(self):
        return self._run("modules/18_outcome_component_sensitivity.py")

    def run_feature_set_sensitivity(self):
        return self._run("modules/19_feature_set_sensitivity.py")

    def run_subgroup_analysis(self):
        return self._run("modules/20_subgroup_analysis.py")

    def run_stratified_thresholds(self):
        return self._run("modules/21_stratified_thresholds.py")

    def run_imbalance_sensitivity(self):
        return self._run("modules/22_imbalance_sensitivity.py")

    def run_survey_weighted_evaluation(self):
        return self._run("modules/23_survey_weighted_evaluation.py")

    def run_permutation_importance(self):
        return self._run("modules/24_permutation_importance.py")

    def finalize_outputs(self):
        return self._run("modules/25_reproducibility_and_master_table.py")

    def get(self, name, default=None):
        return self.namespace.get(name, default)


def banner(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def finish_block(t0):
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f} seconds.")
    return elapsed
