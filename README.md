# Deep Reinforcement Learning based Malicious URL Detection

## Overview
This project develops a web-based application for detecting malicious URLs, leveraging a hybrid approach combining **ALBERT-base-v2** for URL embedding extraction and a **Deep Q-Network (DQN)** for adaptive binary classification (benign vs. phishing). The system achieves **95% accuracy** on a balanced dataset of 1,000 URLs sourced from Cisco Umbrella, URLhaus, and PhishTank. It includes a **Flask-based authentication module** for secure access and a **Streamlit interface** for interactive URL analysis. A custom **Gym environment** facilitates DQN training with a reward system (+1 for correct, -1 for incorrect classifications). The project addresses limitations of static blacklists and supervised models, offering scalability and adaptability for real-time cybersecurity applications.

## Features
- **ALBERT-based Feature Extraction**: Uses ALBERT-base-v2 to generate 768-dimensional URL embeddings.
- **Deep Q-Network (DQN)**: Employs reinforcement learning for adaptive URL classification.
- **Custom Gym Environment**: Simulates URL classification with a reward-based system.
- **Flask Authentication**: Ensures secure user access to the application.
- **Streamlit Interface**: Provides an interactive platform for URL input and real-time analysis.
- **High Accuracy**: Achieves 95% accuracy on a balanced dataset.
- **Scalability**: Modular architecture supports future enhancements like real-time threat feed integration.

## Installation
1. Clone the repository:
```
git clone https://github.com/DVSRBharadwaj/Malicious-URL-Detection.git
```
2. Navigate to the project directory:
```
md Malicious-URL-Detection
```
3. Install the required dependencies:
```
pip install -r requirements.txt
```
4. Ensure **Python 3.7+** is installed, along with dependencies like `torch`, `transformers`, `gym`, `flask`, `streamlit`, `pandas`, and `numpy`.

## Usage
1. **Prepare the Dataset**:
- Place the dataset (CSV or text) in the `data/` folder.
- The project uses a balanced dataset of 1,000 URLs from Cisco Umbrella (benign) and URLhaus/PhishTank (malicious).
- Alternatively, download datasets from [Kaggle](https://www.kaggle.com/datasets) or [PhishTank](https://phishtank.org/).

2. **Preprocess the Data**:
```
python preprocess.py
```
This script processes URLs and extracts embeddings using ALBERT-base-v2.

3. **Train the DQN Model**:
```
python train.py
```
This trains the DQN model within the custom Gym environment.

4. **Run the Web Application**:
- Start the Flask authentication server:
  ```
  python app.py
  ```
- Launch the Streamlit interface:
  ```
  streamlit run interface.py
  ```
- Access the application at `http://localhost:8501` (Streamlit) or `http://localhost:5000` (Flask).

5. **Test a URL**:
- Use the Streamlit interface to input a URL and view the classification result (benign or phishing).
- Alternatively, use the command line:
  ```
  python predict.py --url "http://example.com"
  ```

## Project Structure
- `data/`: Stores datasets or scripts to fetch them.
- `preprocess.py`: Extracts URL embeddings using ALBERT-base-v2.
- `train.py`: Trains the DQN model in the custom Gym environment.
- `predict.py`: Classifies URLs as benign or phishing.
- `app.py`: Flask-based authentication server.
- `interface.py`: Streamlit-based user interface.
- `models/`: Saves trained models.
- `requirements.txt`: Lists required Python packages.

## Dependencies
- Python 3.7+
- torch
- transformers (for ALBERT-base-v2)
- gym
- flask
- streamlit
- pandas
- numpy
- scikit-learn

## Dataset
The system uses a balanced dataset of 1,000 URLs:
- **Benign URLs**: Sourced from Cisco Umbrella.
- **Malicious URLs**: Sourced from URLhaus and PhishTank, covering phishing, malware, spam, and defacement threats.
- Optional sources: [Kaggle URL datasets](https://www.kaggle.com/datasets) or [Malware Domain Blacklist](https://www.malwaredomainlist.com/).

## Model Performance
- **Accuracy**: 95% on a balanced dataset of 1,000 URLs.
- **Classification**: Binary (benign vs. phishing), with a rule-based classifier for specific threat types (e.g., malware, spam).
- **Features**:
- ALBERT-base-v2 generates 768-dimensional URL embeddings.
- DQN adapts to new threats via a reward system (+1 for correct, -1 for incorrect).
- Lightweight design ensures efficient resource usage.

## System Architecture
The system follows a modular pipeline:
1. **Input**: URLs are input via the Streamlit interface or command line.
2. **Preprocessing**: ALBERT-base-v2 extracts embeddings.
3. **Classification**: DQN classifies URLs in a custom Gym environment.
4. **Authentication**: Flask ensures secure access.
5. **Output**: Results are visualized via Streamlit.

## Contributing
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make changes and commit (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a Pull Request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- **Team Member**:
- D.V.S.R. Bharadwaj
- **References**:
- ALBERT: "A Lite BERT for Self-Supervised Learning of Language Representations".
- DQN: "Human-level Control through Deep Reinforcement Learning".
- Datasets: Cisco Umbrella, URLhaus, PhishTank.
