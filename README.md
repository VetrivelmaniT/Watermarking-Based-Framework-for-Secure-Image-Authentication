# 🔐 Watermarking-Based Framework for Secure Image Authentication

A Python-based framework designed to provide secure image authentication through advanced digital watermarking techniques. The project ensures copyright protection and enables tamper localization to detect any unauthorized modifications.

## 📌 Features

- ✅ Dual watermarking: Visible & Invisible
- 🧠 Tamper detection and localization
- 🔍 SSIM and Delta-E based image comparison
- 💾 Extracts and stores changes for forensic use
- 📊 Generates comprehensive output visuals
- 🧰 Built using Python, OpenCV, NumPy, Matplotlib

## 📂 Folder Structure

project/ │ ├── input/ # Folder for input images ├── output/ # Stores comparison and result images ├── watermarking.py # Main logic for watermarking and comparison ├── utils.py # Helper functions (diff, SSIM, Delta-E, etc.) ├── requirements.txt # List of dependencies └── README.md # Project description and guide

## 🛠 Technologies Used

- **Language:** Python 3.x  
- **Libraries:** OpenCV, NumPy, Matplotlib, Scikit-Image

## 📥 Installation

1. Clone this repository:
bash
   git clone https://github.com/VetrivelmaniT/Watermarking-Based-Framework-for-Secure-Image-Authentication
   cd watermark-auth-framework
Install dependencies:


pip install -r requirements.txt
🚀 Usage
Place the original and tampered images in the input/ folder.

Run the main script:

bash
python watermarking.py
Outputs such as difference images, highlighted changes, and watermarked outputs will be saved in the output/ folder.

🧪 Output Examples
01_original.png – Original Image

02_tampered.png – Tampered Image

07_highlighted_diff.png – Tampered regions highlighted

11_comparison_output.png – Side-by-side comparison

📈 Applications
Copyright protection for digital images

Legal document image validation

Tamper localization in security-sensitive systems

📧 Contact
Created by Vetrivel Mani T
Email: tvetrivelmani@gmail.com
Portfolio: https://vetrivel-mani-t-portfolio-com.vercel.app/
