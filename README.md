# Cancer Risk Assistant

An end-to-end AI prototype that combines a self-study Data Science
project with an LLM, RAG, tool routing, and a Gradio chat interface.

The project started as a Data Science analysis of cancer severity and
was later extended into a conversational assistant. The goal was to let
the LLM use deterministic tools rather than generating the severity
score or source information from its own knowledge.

------------------------------------------------------------------------

## 1. Project Overview

The project has two main tools:

1.  **Cancer Severity Prediction** --- uses a Linear Regression model
    from my self-study Data Science project to predict a numerical
    cancer severity score from selected risk factors.
2.  **Guideline Retrieval (RAG)** --- retrieves relevant information
    from official WHO and IARC webpages.

The overall flow is:

``` text
User Question
      |
      v
   Qwen LLM
   Router
      |
      +----------------------+
      |                      |
      v                      v
Prediction Tool          RAG Tool
      |                      |
      |                      |
      +----------+-----------+
                 |
                 v
             Tool Results
                 |
                 v
            Qwen LLM
             Composer
                 |
                 v
           Final Response
```

The LLM does not calculate the severity score itself. Python runs the
tools and provides the results to the LLM for the final response.

------------------------------------------------------------------------

# Part 1 --- Data Science Foundation

## 2. The Original Data Science Project

The project started as a self-study Data Science project using the
**Global Cancer Patients 2015--2024** dataset from Kaggle.

The main question was:

> **How much do personal, behavioral, and environmental factors
> influence the severity of cancer in patients?**

The dataset contains **50,000 samples and 15 columns**, including
demographic information, lifestyle factors, environmental factors,
cancer information, treatment cost, survival years, and a target
severity score.

The main features selected for the prediction model were:

-   Age
-   Genetic_Risk
-   Air_Pollution
-   Alcohol_Use
-   Smoking
-   Obesity_Level

The target was:

-   `Target_Severity_Score`

### Dataset

Source: [Global Cancer Patients 2015--2024 ---
Kaggle](https://www.kaggle.com/datasets/zahidmughal2343/global-cancer-patients-2015-2024)

------------------------------------------------------------------------

## 3. Data Cleaning

I first checked the dataset for missing values and unexpected data.

The dataset contained **50,000 non-null values for all 15 columns**, so
no rows needed to be removed or filled because of missing values.

This allowed me to continue with the selected features without applying
additional missing-value treatment.

------------------------------------------------------------------------

## 4. Data Pre-processing

I selected six features:

``` text
Age
Genetic_Risk
Air_Pollution
Alcohol_Use
Smoking
Obesity_Level
```

`Target_Severity_Score` was used as the target variable.

The data was split into:

-   **80% training data**
-   **20% testing data**

using `random_state=42`.

I also used `StandardScaler` to standardise the input features before
modelling.

------------------------------------------------------------------------

## 5. Exploratory Data Analysis

I used a correlation heatmap to understand the relationships between the
selected variables and the target.

The main observations were:

-   `Genetic_Risk` had a moderate positive correlation with severity.
-   `Smoking` had a moderate positive correlation with severity.
-   `Air_Pollution`, `Alcohol_Use`, and `Obesity_Level` had weaker
    positive correlations.
-   `Age` had almost no correlation with the target.

The correlation results suggested that Genetic Risk and Smoking were the
strongest relationships with the target in this dataset.

------------------------------------------------------------------------

## 6. Model Selection

I compared **Linear Regression** and **Random Forest** models.

I started with Linear Regression because it is easy to understand and
allows the relationship between each feature and the predicted score to
be examined.

I then tested Random Forest because it can capture more complicated
patterns.

After comparing the models, I selected **Linear Regression** for the
final tool because it achieved better performance on the test set.

### Linear Regression performance

-   MSE: **0.305**
-   MAE: **0.480**
-   R²: **0.785**

The model explains around **78.5% of the variance in the target severity
score within this dataset**.

------------------------------------------------------------------------

## 7. Turning the Data Science Model into a Tool

The next step was to reuse the trained Data Science model inside the
chatbot instead of training a new model for every question.

The trained model is saved as:

``` text
severity_model.joblib
```

The model is a pipeline containing:

``` text
StandardScaler
      |
      v
Linear Regression
```

The chatbot can therefore load the same trained model and use it
directly.

### Why use Joblib?

`joblib` allows the trained Python model to be saved and loaded later.

This means the LLM project can reuse the model from the Data Science
project without retraining it every time.

------------------------------------------------------------------------

## 8. Building the Prediction Tool

I wrapped the model inside a callable function:

``` text
predict_severity(...)
```

The tool does more than simply call `model.predict()`.

It:

1.  Accepts the risk factors extracted from the user's question.
2.  Uses dataset averages for missing factors.
3.  Records which factors were filled automatically.
4.  Predicts the severity score.
5.  Calculates the contribution of each factor relative to its dataset
    mean.
6.  Runs a counterfactual prediction when smoking is above the dataset
    mean, using a scenario where smoking is reduced to zero.

This gives the LLM structured information that it can use to explain the
model output.

------------------------------------------------------------------------

# Part 2 --- Extending the Project with an LLM

## 9. Why Add an LLM?

The original Data Science project required the user to provide inputs
directly to a Python function.

I wanted to make the project more interactive by allowing a user to ask
questions naturally.

For example:

``` text
I'm 55, I smoke heavily and my father had cancer. How risky am I?
```

Instead of manually calling the prediction function, the LLM can
understand the question, extract the relevant information, and decide
which tool should be used.

This changes the project from a standalone prediction model into a
conversational AI system.

------------------------------------------------------------------------

## 10. Choosing the LLM

I initially planned to use Qwen through an external API, such as
OpenRouter. However, this would require an API key and users would need
to sign in or provide their own credentials to run the chatbot.

I was concerned that this would make the project less convenient for
someone who wants to download and try the notebook.

I therefore chose **Qwen2.5-1.5B-Instruct** and load the model directly
into the Google Colab environment.

I previously tried other open-source LLMs such as LLaMA and local model
serving with Ollama during an academic project, but I found that Qwen
performed better for my use case.

The model runs locally on the available Colab GPU, so no external LLM
API or API key is required.

------------------------------------------------------------------------

# Part 3 --- Adding RAG

## 11. Guideline Retrieval Tool

The second tool was added because the prediction model alone cannot
answer general cancer information questions.

For example:

``` text
Does alcohol really cause cancer?
```

This question does not require the prediction model. It requires
reliable public-health information.

I therefore added a Retrieval-Augmented Generation (RAG) tool using
information from official **WHO and IARC webpages**.

The current knowledge base contains seven sources:

-   WHO Cancer fact sheet
-   WHO Tobacco fact sheet
-   WHO Alcohol fact sheet
-   WHO Obesity and overweight fact sheet
-   WHO Ambient air pollution fact sheet
-   WHO Physical activity fact sheet
-   IARC Monographs overview

The source URLs are stored in the notebook and the webpage content is
downloaded directly from those sources.

### Sources

-   [WHO Cancer fact
    sheet](https://www.who.int/news-room/fact-sheets/detail/cancer)
-   [WHO Tobacco fact
    sheet](https://www.who.int/news-room/fact-sheets/detail/tobacco)
-   [WHO Alcohol fact
    sheet](https://www.who.int/news-room/fact-sheets/detail/alcohol)
-   [WHO Obesity and overweight fact
    sheet](https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight)
-   [WHO Ambient air pollution fact
    sheet](https://www.who.int/news-room/fact-sheets/detail/ambient-(outdoor)-air-quality-and-health)
-   [WHO Physical activity fact
    sheet](https://www.who.int/news-room/fact-sheets/detail/physical-activity)
-   [IARC Monographs](https://monographs.iarc.who.int/)

------------------------------------------------------------------------

## 12. Chunking and Embedding

The webpage content is processed into retrieval units while keeping the
original document name and URL.

I use:

``` text
all-MiniLM-L6-v2
```

as the embedding model.

Embedding converts text into numerical vectors so that the system can
compare the meaning of a question with the stored information.

For example, a question such as:

``` text
How do I lower my risk?
```

can be matched with information about prevention even if the exact words
are different.

The embedding model is small and free, so it is practical for the Colab
environment.

If the embedding model cannot be loaded, the notebook has a TF-IDF
fallback.

------------------------------------------------------------------------

## 13. Retrieval

For this project, the knowledge base is small, so I use cosine
similarity with a NumPy array instead of a vector database.

The current relevance cutoff is:

``` text
0.25
```

for the embedding-based retrieval.

If no retrieved result reaches the relevance threshold, the tool returns
no result instead of forcing the system to use the closest match.

This is intentional because using an unrelated source would be worse
than returning no answer.

A vector database would make more sense if the knowledge base grows
significantly, for example to tens of thousands of chunks.

------------------------------------------------------------------------

# Part 4 --- Connecting the LLM to the Tools

## 14. The Router

Once the two tools were available, the next problem was deciding which
tool should be called for each user question.

I added a **Router** using Qwen.

The LLM receives the available tools as JSON schemas and is asked to
return a JSON object containing:

``` text
tool name
+
arguments
```

For example:

``` json
{
  "calls": [
    {
      "tool": "predict_severity",
      "args": {
        "age": 55,
        "smoking": 8,
        "genetic_risk": 7
      }
    }
  ],
  "status": "ok"
}
```

The router can select:

``` text
predict_severity
```

or:

``` text
search_guidelines
```

It can also select both tools when the question needs both prediction
and guideline information.

------------------------------------------------------------------------

## 15. Why JSON Instead of Native Function Calling?

I chose JSON because it makes the routing logic simple and easy to see
in the notebook.

The LLM first decides what the user is asking for and then returns the
tool name and arguments in a structured format.

The trade-off is that the JSON output needs to be checked and validated
before a tool is called.

I therefore added validation and a rule-based fallback router.

If Qwen produces invalid JSON or incomplete arguments, the fallback uses
keyword matching to select the appropriate tool.

The fallback is less flexible than the LLM router, but it keeps the rest
of the pipeline runnable.

------------------------------------------------------------------------

# Part 5 --- Execute and Compose

## 16. Executing the Tool Calls

The Router only decides what should happen.

It does not execute the tools.

Python handles the execution:

``` text
Router
   |
   v
Execute
   |
   +--> Prediction Tool
   |
   +--> RAG Tool
```

This keeps the actual tool results deterministic and separate from the
LLM's response generation.

A guard was also added to avoid producing a severity score when too few
personal details are available.

For example, a general question about quitting smoking should not
accidentally become a personalised prediction using mostly dataset
averages.

------------------------------------------------------------------------

## 17. Composing the Final Reply

After the tools return their results, the results are passed back to
Qwen.

Qwen is only responsible for turning the tool output into readable
English.

The system separates responsibilities:

``` text
Python
  |
  +--> Severity score
  +--> Assumed inputs
  +--> Source links
  |
  v
Qwen
  |
  +--> Explanation
```

This means the severity score comes from the regression model and the
source links come from the retrieval tool.

The LLM is not responsible for calculating the score or creating the
source URLs.

------------------------------------------------------------------------

# Part 6 --- Chat Interface

## 18. Gradio Interface

I use **Gradio** to create a chat interface that runs directly in Google
Colab.

The interface allows users to interact with the chatbot without manually
running the Python functions.

The interface includes:

-   Chat history
-   Example questions
-   Text input
-   Send button
-   Clickable source links
-   Educational disclaimer

The notebook uses `share=True` to create a temporary public Gradio link
for demonstrations.

------------------------------------------------------------------------

# Part 7 --- Evaluation

## 19. Evaluation Strategy

I use two tests to check whether the main parts of the system are
working properly.

### 19.1 Numeric Fidelity

I tested **10 cases** to check whether the assistant reports the model's
score correctly.

The model's original score is compared with the score reported in the
assistant response.

Result:

``` text
Numeric fidelity: 10/10 passed
```

This checks that the severity score is not changed by the LLM.

### 19.2 Retrieval Accuracy

I tested **10 questions** where I already knew which source should
contain the relevant information.

Result:

``` text
Retrieval accuracy: 6/10 passed
```

The test also includes two negative cases where the correct behaviour is
to return no result.

This test focuses on whether the retriever finds the expected source. It
does not test whether the LLM uses the retrieved information correctly
in its final response.

------------------------------------------------------------------------

# Part 8 --- End-to-End Results

## 20. Chat Interface Testing

I also manually tested the complete system through the Gradio interface.

Eight questions were tested across different routing paths.

### Results

-   **5/8 cases passed completely**
-   **3/8 cases were partial**

The partial cases showed three different limitations:

1.  The LLM added unsupported statistics to a response.
2.  The router incorrectly treated "quitting smoking" as a personal
    smoking input.
3.  The LLM incorrectly interpreted a negative contribution value.

Importantly, the severity score itself remained consistent with the
prediction tool in the tested cases.

------------------------------------------------------------------------

# Part 9 --- Limitations

## 21. Small LLM

The composer and router use **Qwen2.5-1.5B-Instruct** so the project can
run end to end on a free Colab T4 GPU.

The small model does not always follow long instruction sets reliably.

This caused the three partial cases observed during end-to-end testing.

### Unsupported information

In one case, the model added statistics that were not present in the
retrieved passages.

This shows that even when RAG retrieves relevant information, a small
LLM can still add information from its own learned knowledge.

### Imperfect routing

In another case, the router saw the word "smoking" in "quitting smoking"
and selected the prediction tool even though the question was not asking
for a personal risk prediction.

A guard prevented a meaningless score from being shown, but the question
was still routed incorrectly.

### Signed contribution values

The model also incorrectly interpreted a negative contribution as
increasing risk.

A contribution of:

``` text
-1.01
```

lowers the model's score, but the LLM described it as increasing risk.

------------------------------------------------------------------------

## 22. Small Knowledge Base

The current knowledge base contains only a small number of WHO and IARC
webpages.

This limits the range of questions that the RAG system can answer.

The retrieval system intentionally returns no result when the similarity
score is below the relevance threshold rather than forcing a weak match.

This reduces the chance of using an unrelated source, but it also
reduces coverage.

------------------------------------------------------------------------

## 23. Synthetic Dataset

The regression model achieves:

``` text
R² = 0.785
```

on held-out data.

However, the training data is a synthetic Kaggle dataset.

Therefore, the severity score should be understood as a prediction
produced from this dataset, not as a clinically validated cancer risk
score.

The model demonstrates the Data Science and tool-calling workflow rather
than providing a clinical prediction system.

------------------------------------------------------------------------

# Part 10 --- What I Built

## 24. From Data Science to AI System

The project developed in stages:

``` text
Stage 1
Data Science
    |
    v
Analyse cancer severity factors
    |
    v
Train and compare ML models
    |
    v
Select Linear Regression
    |
    v
Save trained model
    |
    v
Stage 2
AI Tool
    |
    v
Wrap the model as predict_severity()
    |
    v
Stage 3
RAG
    |
    v
Collect WHO/IARC sources
    |
    v
Embed and retrieve relevant information
    |
    v
Stage 4
LLM
    |
    v
Use Qwen to understand user questions
    |
    v
Route questions to tools
    |
    v
Stage 5
Tool Calling
    |
    v
Execute tools with Python
    |
    v
Send results back to Qwen
    |
    v
Generate final response
    |
    v
Stage 6
Application
    |
    v
Build Gradio chat interface
```

The main idea was not to replace the Data Science model with an LLM.

Instead, I used the LLM as the conversational layer around existing
deterministic tools.

------------------------------------------------------------------------

# 25. Key Technologies

  Area               Technology
  ------------------ ---------------------------
  Programming        Python
  Data analysis      Pandas, NumPy
  Machine Learning   Scikit-learn
  Prediction model   Linear Regression
  Model comparison   Random Forest
  Model saving       Joblib
  LLM                Qwen2.5-1.5B-Instruct
  LLM framework      Hugging Face Transformers
  Embeddings         all-MiniLM-L6-v2
  Retrieval          Cosine similarity + NumPy
  RAG sources        WHO and IARC
  Interface          Gradio
  Environment        Google Colab / T4 GPU

------------------------------------------------------------------------

# 26. Repository Structure

A simple structure for the GitHub repository:

``` text
Cancer-Severity-Project/
│
└── Cancer Severity Project/
    │
    ├── Cancer Risk Assistant Test Cases/
    │   ├── Test Case 1.1.png
    │   ├── Test Case 1.2.png
    │   ├── Test Case 2.1.png
    │   ├── Test Case 2.2.png
    │   ├── Test Case 3.png
    │   ├── Test Case 4.png
    │   └── Test Case 5.png
    │
    ├── Cancer_Risk_Assistant_Project.ipynb
    ├── Cancer_Severity_Prediction_Project.ipynb
    ├── global_cancer_patients_2015_2024.csv
    ├── severity_model.joblib
    └── README.md
```

------------------------------------------------------------------------

# 27. Running the Project

The LLM version is designed to run in Google Colab.

### Requirements

-   Python
-   Google Colab
-   GPU runtime
-   T4 GPU recommended
-   Internet access for downloading models and WHO/IARC webpages

### Install dependencies

``` bash
pip install -q sentence-transformers gradio scikit-learn joblib pypdf transformers accelerate bitsandbytes
```

### GPU

The notebook checks that a GPU is available before loading the LLM.

In Google Colab:

``` text
Runtime
  >
Change runtime type
  >
T4 GPU
```

No external LLM API key is required.

------------------------------------------------------------------------

# 28. Disclaimer

This project is an **educational AI prototype**.

The severity model is trained on a synthetic Kaggle dataset and is not
clinically validated. The severity score should not be interpreted as a
real patient's probability of developing cancer or as a medical
diagnosis.

The guideline information is retrieved from official WHO and IARC
sources.

The system is designed to demonstrate:

-   Data Science
-   Machine Learning
-   LLM integration
-   RAG
-   Tool routing
-   Deterministic tool execution
-   AI application architecture
