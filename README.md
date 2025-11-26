# halegannada-to-english-transalation-using-seq2seqmodel

halegannada-to-english-transalation-using-seq2seqmodel

Clone the project and open the project folder in vs code

open terminal and enter cd "add the path of server.py file"

python server.py

in server.py -> add geminai api key and deepseek api keys

geminai api key -> speaker functionality-> mutilanguage
deepseek Api key -> text transaltion feature

https://api.deepseek.com

For seq to seq model testng:

test data present in ->data folder -> db.csv file
test data for json page -> data folder -> db.json file

# Halegannada Translation Hub

## Project Summary

The Halegannada Translation Hub is a web application designed to bridge the gap between ancient Kannada literature and modern readers. It provides tools to translate Halegannada (Old Kannada) text and even entire images containing Halegannada script into modern English. This project leverages advanced AI and machine learning models to provide accurate and context-aware translations, making centuries of wisdom and literature accessible to everyone.

## Technologies Used

- **Backend:**
  - Python
  - Flask
  - Gunicorn (for production)
- **Frontend:**
  - HTML
  - CSS
  - JavaScript
  - Bootstrap
- **Machine Learning & AI:**
  - Google Gemini Pro: For Optical Character Recognition (OCR) to extract text from images.
  - DeepSeek API (via OpenRouter): For translating Halegannada text to English.
  - scikit-learn (Pickle): For loading the Halegannada to modern Kannada dictionary.

## Steps to Run the Project

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/halegannada-to-english-transalation.git
    cd halegannada-to-english-transalation
    ```

2.  **Create a virtual environment and activate it:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up API Keys:**

    - The application requires API keys for Google Gemini and DeepSeek (via OpenRouter). These keys are currently hardcoded in `server.py`. For a production environment, it is highly recommended to use environment variables.

5.  **Run the Flask development server:**
    ```bash
    python server.py
    ```
    The application will be available at `http://127.0.0.1:5000`.

## Steps to Make Changes to the Project

1.  **Follow the steps above to set up the project locally.**

2.  **Make your desired changes to the codebase.**

    - **Backend:** The main application logic is in `server.py`. This is where you can modify API integrations, translation logic, and endpoints.
    - **Frontend:** The user interface files are in the `pages/` directory and `index.html`. You can modify these files to change the look and feel of the application.
    - **Static Assets:** The `assets/` directory contains images, stylesheets, and the `Dictionary.pkl` file.

3.  **Test your changes locally** by running the development server and interacting with the application.

4.  **Once you are satisfied with your changes, you can commit them to your forked repository and create a pull request.**
