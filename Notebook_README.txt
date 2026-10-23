Predictive Public Infrastructure Maintenance
================================================

FINAL MODEL PACKAGE

Contents
--------
best.pt
    Trained YOLO detector weights.

class_names.json
    Detector class IDs and damage names.

inference_config.json
    Image size, confidence threshold and IoU threshold.

risk_config.json
    Maintenance risk and urgency policy parameters.

output_schema.json
    Expected detection, risk and inspection outputs.

model_metadata.json
    Complete model and configuration metadata.

Architecture
------------
Input Image
    |
    v
YOLO Damage Detector
    |
    v
Damage Detections
    |
    v
Risk / Severity Calculation
    |
    v
Maintenance Priority
    |
    v
Inspection Queue

Important
---------
The trained detector and maintenance policy are separate.

best.pt contains the learned model parameters.

risk_config.json contains the prototype maintenance
decision rules.

Changing maintenance thresholds does not require retraining
the detector.

The current RDD2022 dataset does not provide real GPS
coordinates for this prototype. Therefore location/zone
aggregation should be replaced with actual GPS or road-segment
metadata in a deployment system.

The maintenance urgency windows are prototype decision rules
and are not engineering-certified repair deadlines.
