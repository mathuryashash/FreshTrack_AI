import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfDriftedColumns, TestShareOfDriftedColumns
import os

class MonitoringService:
    def __init__(self, reference_data_path):
        self.reference_data_path = reference_data_path
        self.reference_data = None
        if os.path.exists(reference_data_path):
            self.reference_data = pd.read_csv(reference_data_path)
            # Standardize columns if necessary
            # For this project, reference data should have: freshness, quality_grade, shelf_life_days, storage_temp, storage_humidity
        
    def generate_drift_report(self, current_data: pd.DataFrame, report_path: str = "reports/drift_report.html"):
        """
        Generates an Evidently HTML report comparing reference data to current data.
        """
        if self.reference_data is None:
            return {"error": "Reference data not found"}
        
        # Ensure common columns match
        common_cols = list(set(self.reference_data.columns) & set(current_data.columns))
        if not common_cols:
            return {"error": "No common columns for drift detection"}
            
        report = Report(metrics=[
            DataDriftPreset(),
            TargetDriftPreset()
        ])
        
        report.run(reference_data=self.reference_data[common_cols], 
                   current_data=current_data[common_cols])
        
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        report.save_html(report_path)
        return {"status": "success", "report_path": report_path}

    def run_drift_tests(self, current_data: pd.DataFrame, threshold: float = 0.2):
        """
        Runs tests to detect if drift exceeds a certain threshold.
        """
        if self.reference_data is None:
            return {"error": "Reference data not found"}

        common_cols = list(set(self.reference_data.columns) & set(current_data.columns))
        
        test_suite = TestSuite(tests=[
            TestShareOfDriftedColumns(lt=threshold)
        ])
        
        test_suite.run(reference_data=self.reference_data[common_cols], 
                       current_data=current_data[common_cols])
        
        return {
            "drift_detected": not test_suite.was_successful(),
            "summary": test_suite.json()
        }

if __name__ == "__main__":
    # Example usage
    monitor = MonitoringService("data/train.csv")
    # Simulate current data
    dummy_current = pd.read_csv("data/train.csv").sample(10)
    # Add some noise to simulate drift
    dummy_current['storage_temp'] = dummy_current['storage_temp'] + 5 
    
    report_info = monitor.generate_drift_report(dummy_current)
    print(f"Report generated: {report_info}")
    
    test_info = monitor.run_drift_tests(dummy_current)
    print(f"Drift test success: {not test_info['drift_detected']}")
