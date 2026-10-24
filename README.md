# 🛣️ MargRakshak — Safer Roads. Stronger India.

**MargRakshak** is an AI-powered road damage detection and reporting platform designed to help identify road defects, visualize reported damage locations on a map, and support faster road maintenance planning.

The project combines computer vision with location-based reporting to help authorities and citizens work toward safer roads and better infrastructure management.

## 🚀 Features

- **AI-Based Damage Detection:** Uses a YOLO object detection model to identify road damage from uploaded images.
- **Road Damage Classification:** Detects categories such as longitudinal cracks, transverse cracks, alligator cracks, other road damage, and potholes, depending on model predictions.
- **Location-Based Reporting:** Allows users to submit road damage reports with location information.
- **Interactive Map:** Displays reported road damage locations using an interactive map.
- **Report Tracking:** Helps users view and track submitted reports.
- **Admin Dashboard:** Provides an interface for administrators to review and manage reports.
- **User-Friendly Interface:** Built with Streamlit for easy access through a web browser.

## 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application interface |
| YOLO / Ultralytics | Road damage detection |
| OpenCV | Image processing |
| Pillow | Image handling |
| NumPy and Pandas | Data processing |
| Folium | Interactive maps |
| Streamlit-Folium | Map integration |
| SQLite | Local report storage |

## 📂 Project Structure

```text
MargRakshak/
│
├── app_redesigned.py    # Main Streamlit application
├── requirements.txt     # Python dependencies
├── best2.pt             # Trained YOLO model weights
├── README.md             # Project documentation
└── assets/               # Optional images and other resources
```
🔄 How It Works

A user uploads an image of a damaged road.

The trained YOLO model analyzes the image and predicts road damage.

The user provides the relevant location information.

The application creates or stores a road damage report.

Reports can be displayed on an interactive map.

Administrators can review reports and update their status.

The collected reports can support infrastructure monitoring and maintenance prioritization.

🎯 Project Objective

The primary objective of MargRakshak is to explore how artificial intelligence and geospatial visualization can improve road damage reporting and maintenance planning.

By combining automated image analysis, location-based reports, and an interactive dashboard, the project aims to make road damage information easier to access and manage.

🔮 Future Scope

Integration with live GPS and mobile reporting.

Improved pothole and crack detection accuracy.

Automated damage severity assessment.

Risk-based maintenance prioritization.

Notifications for critical road damage.

Integration with government road maintenance systems.

Cloud database integration for persistent reports.

Support for regional languages across India.

Analytics dashboards for identifying damage hotspots.

⚠️ Limitations

Detection results depend on the quality and diversity of the training data.

Predictions may contain false positives or missed road defects.

Location accuracy depends on the information submitted by users.

AI predictions should be verified before official maintenance decisions are made.

Deployment performance depends on the available computing resources.
