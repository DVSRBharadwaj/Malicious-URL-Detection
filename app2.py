import streamlit as st
import torch
import numpy as np
from transformers import AlbertTokenizer, AutoModel
from stable_baselines3 import DQN
import gym
from gym import spaces
import os
import pandas as pd
import re
import sqlite3

# Main app
def main():
    # Set page config as the first Streamlit command
    st.set_page_config(page_title="Malicious URL Detector", layout="wide")

    # Function to get the logged-in username from the flag file
    def get_logged_in_username():
        flag_file = "login_success.flag"
        if os.path.exists(flag_file):
            with open(flag_file, "r") as f:
                username = f.read().strip()
            return username
        return None

    # Function to classify phishing type
    def classify_phishing_type(url):
        url = url.lower()
        # Malware: URLs with executable or archive extensions, download keywords
        if re.search(r'\.exe|\.zip|\.rar|\.apk|download', url):
            return "Malware"
        # Spam: URLs with login, verification, or account-related keywords
        elif re.search(r'login|verify|account|signin|signup|password', url):
            return "Spam"
        # Defacement: URLs with hack, deface, or random strings (simplified heuristic)
        elif re.search(r'hack|deface|[0-9a-f]{10,}', url):
            return "Defacement"
        # Default: Generic phishing if no specific pattern matches
        return "Generic Phishing"

    # Custom URLEnvironment for prediction
    class URLEnvironment(gym.Env):
        def __init__(self, model_dir):
            super(URLEnvironment, self).__init__()
            try:
                self.tokenizer = AlbertTokenizer.from_pretrained(model_dir, local_files_only=True)
                self.model = AutoModel.from_pretrained(model_dir, local_files_only=True)
                self.action_space = spaces.Discrete(2)  # 0: benign, 2: phishing
                self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(768,), dtype=np.float32)
                self.label_to_action = {0: 0, 2: 1}
                self.action_to_label = {0: 0, 1: 2}
            except Exception as e:
                raise ValueError(f"Failed to initialize URLEnvironment: {e}")

        def get_embedding(self, url):
            inputs = self.tokenizer(url, return_tensors="pt", max_length=128, truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    # Load models
    @st.cache_resource
    def load_models():
        try:
            model_dir = "./albert-base-v2"
            dqn_model_path = "./drl_url_detector.zip"
            if not os.path.exists(model_dir) or not os.path.exists(dqn_model_path):
                raise FileNotFoundError(f"Model directory {model_dir} or DQN model {dqn_model_path} not found")
            env = URLEnvironment(model_dir)
            dqn_model = DQN.load(dqn_model_path)
            return env, dqn_model
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return None, None

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "About", "Detect"])

    env, dqn_model = load_models()
    if env is None or dqn_model is None:
        st.error("Failed to load models. Please check model files and paths.")
        return

    label_map = {0: "Benign", 2: "Phishing"}

    if page == "Home":
        st.title("Malicious URL Detector")
        # Display welcome message with username
        username = get_logged_in_username()
        if username:
            st.write(f"Welcome back, {username}!")
        else:
            st.write("Welcome to the Malicious URL Detector! Please log in to continue.")
        st.write("""
            This project uses Deep Reinforcement Learning to classify URLs as Benign or Phishing (including malware, spam, and defacement threats). Navigate to the Detect page to check a URL's safety.
        """)

    elif page == "About":
        st.title("About the Project")
        st.write("""
            ### Project Overview
            The Malicious URL Detector leverages a Deep Q-Network (DQN) combined with a Transformer model (ALBERT-base-v2) to classify URLs into two categories: Benign and Phishing. Phishing encompasses threats like malware, spam, and defacement URLs. The model was trained to achieve 95% accuracy in distinguishing safe URLs from malicious ones.

            ### Models
            #### ALBERT-base-v2
            - **Creator**: Google Research
            - **Publication Year**: 2019
            - **Description**: ALBERT (A Lite BERT) is a lightweight version of BERT, designed for efficiency with fewer parameters while maintaining performance. It uses factorized embedding parameterization and cross-layer parameter sharing to reduce memory usage.
            - **Architecture**: 12 transformer layers, 12 attention heads, hidden size of 768, resulting in approximately 12 million parameters (compared to BERT-base’s 110 million).
            - **Performance**: On general NLP benchmarks (e.g., GLUE, SQuAD), ALBERT-base-v2 achieves near state-of-the-art results with reduced computational cost. In this project, it generates 768-dimensional embeddings for URLs, capturing semantic and structural features.
            - **Training**: Pre-trained on a large corpus (e.g., Wikipedia, BookCorpus) using masked language modeling and sentence order prediction objectives.
            - **Source**: Available via Hugging Face Transformers library (https://huggingface.co/albert-base-v2).
            - **Use in Project**: Extracts URL embeddings as input to the DQN, enabling robust feature representation for classification.

            #### Deep Q-Network (DQN)
            - **Creator**: DeepMind
            - **Publication Year**: 2013 (initial paper), 2015 (Nature paper with significant improvements)
            - **Description**: DQN is a reinforcement learning algorithm that combines Q-learning with deep neural networks to handle high-dimensional state spaces. It uses experience replay and target networks for stable training.
            - **Architecture**: In this project, the DQN uses a multilayer perceptron (MLP) policy with two hidden layers (64 units each, ReLU activation), mapping 768-dimensional URL embeddings to two actions (Benign, Phishing).
            - **Performance**: In the 2015 Atari benchmark, DQN achieved human-level performance on 49 games. In this project, it achieves 95% accuracy on a 1,000-URL dataset, with stable convergence after 10,000 timesteps.
            - **Training**: Trained using a replay buffer (size 10,000), batch size of 32, learning rate of 0.001, and a custom Gym environment (URLEnvironment) with rewards (+1 for correct, -1 for incorrect classification).
            - **Source**: Implemented via Stable-Baselines3 (https://stable-baselines3.readthedocs.io).
            - **Use in Project**: Classifies URLs based on ALBERT embeddings, outputting Benign (0) or Phishing (2) with confidence scores derived from Q-values.
        """)
        # Display the performance comparison image with a custom width
        st.image("chart.png", caption="Performance Comparison of URL Detection Models", width=600)

        st.write("""
            ### Dataset
            - **Size**: 1,000 URLs (500 Benign, 500 Phishing).
            - **Sources**:
              - **Benign URLs**: Sourced from Cisco Umbrella Top 1M, representing safe, popular websites.
              - **Phishing URLs**: Collected from PhishTank and OpenPhish, including malware, spam, and defacement threats.
            - **Preprocessing**: URLs were cleaned (e.g., removing query parameters) and balanced to ensure equal representation of both classes.

            ### Methodology
            - **Feature Extraction**: ALBERT-base-v2 generates embeddings for each URL, which serve as the state input for the DQN.
            - **Environment**: A custom Gym environment (URLEnvironment) provides rewards (+1 for correct classification, -1 for incorrect) based on URL labels.
            - **Training**: The DQN was trained for 10,000 timesteps using a replay buffer (size 10,000), batch size of 32, and learning rate of 0.001. The agent achieved 95% accuracy.
            - **Prediction**: The trained DQN predicts Benign (0) or Phishing (2) with a confidence score derived from softmaxed Q-values. For phishing URLs, a rule-based classifier identifies the specific threat type (malware, spam, defacement).

            ### Usage
            On the Detect page, enter a URL to check its safety. The system will classify it as Benign or Phishing, display a confidence percentage, and identify the phishing type (if applicable). A redirect option is provided for safe URLs, while access is restricted for phishing URLs with over 80% confidence.

            ### Limitations
            - Limited to 1,000 URLs in training, which may not capture all malicious patterns.
            - Phishing label includes diverse threats (malware, spam, defacement), potentially reducing specificity.
            - Offline mode requires pre-downloaded model files.
            - Phishing type detection uses heuristic rules, which may misclassify complex URLs.

            ### Future Improvements
            - Expand the dataset to include more URLs and diverse threat types with sub-labels.
            - Fine-tune ALBERT for URL-specific features.
            - Train a multiclass DQN to directly predict malware, spam, and defacement.
            - Integrate real-time threat feeds (e.g., Google Safe Browsing API) for dynamic updates.
        """)

    elif page == "Detect":
        st.title("Detect Malicious URLs")
        url_input = st.text_input("Enter URL to check (e.g., http://example.com)", "")
        if st.button("Detect"):
            if url_input:
                try:
                    embedding = env.get_embedding(url_input)
                    action, _ = dqn_model.predict(embedding, deterministic=True)
                    action = int(action)
                    q_values = dqn_model.q_net(torch.tensor([embedding], dtype=torch.float32)).detach().numpy()[0]
                    probabilities = np.exp(q_values) / np.sum(np.exp(q_values))
                    confidence = probabilities[action] * 100
                    url_type = label_map[env.action_to_label[action]]
                    
                    st.write(f"**Prediction**: {url_type}")
                    st.write(f"**Confidence**: {confidence:.2f}%")
                    if url_type == "Phishing":
                        phishing_type = classify_phishing_type(url_input)
                        st.write(f"**Phishing Type**: {phishing_type}")

                    st.session_state.url_result = {
                        'url': url_input,
                        'type': url_type,
                        'confidence': confidence,
                        'action': action,
                        'phishing_type': phishing_type if url_type == "Phishing" else None
                    }

                except Exception as e:
                    st.error(f"Error processing URL: {e}")
            else:
                st.warning("Please enter a URL")

        if 'url_result' in st.session_state:
            result = st.session_state.url_result
            if result['type'] == "Phishing" and result['confidence'] > 80:
                st.session_state.redirect = True
                st.experimental_rerun()
            else:
                if st.button("Redirect to URL"):
                    st.markdown(f'<meta http-equiv="refresh" content="0;url={result["url"]}">', unsafe_allow_html=True)

    if 'redirect' in st.session_state and st.session_state.redirect:
        st.title("Access Restricted")
        st.error("Malicious website")
        if st.button("Back to Detect"):
            st.session_state.redirect = False
            st.session_state.url_result = None
            st.experimental_rerun()

if __name__ == '__main__':
    main()