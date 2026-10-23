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
