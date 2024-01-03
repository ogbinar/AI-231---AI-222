import pandas as pd
import time
import re
import os
import nltk
from nltk.util import ngrams
from nltk.metrics import jaccard_distance
from llama_cpp import Llama
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain

# Set HuggingFaceHub API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Function to calculate similarity scores
def calculate_similarity_score(reference, candidate):
    # Tokenize the strings into words
    reference = str(reference)
    candidate = str(candidate)
    
    tokens1 = nltk.word_tokenize(reference)
    tokens2 = nltk.word_tokenize(candidate)

    # Exact Match Score
    exact_match_score = 1.0 if tokens1 == tokens2 else 0.0

    # N-Gram Matching Score
    n = 2  # Define the n-gram size (bi-grams in this case)
    ngrams1 = list(ngrams(tokens1, n))
    ngrams2 = list(ngrams(tokens2, n))

    # Calculate Jaccard similarity for N-Gram matching
    ngram_match_score = jaccard_distance(set(ngrams1), set(ngrams2))

    # em = 1 is exact
    # ngram = 1 is exact
    return exact_match_score, 1 - ngram_match_score  # Return the scores

# Read the Excel file
df = pd.read_excel("MAROON CHAT Q&A.xlsx", sheet_name="Public")

# Define model path
model_path = "merged_model/ggml-model-Q4_0.gguf"

# Initialize LlamaCpp
llm = LlamaCpp(
    model_path=model_path,
    temperature=0.1,
    max_tokens=256,
    n_gpu_layers=10000,
    top_p=1,
    n_ctx=4096,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
)

# Define HuggingFaceHub repo ID
repo_id = "HuggingFaceH4/zephyr-7b-beta"

# Initialize HuggingFaceHub
llm_judge = HuggingFaceHub(
    repo_id=repo_id,
    model_kwargs={"temperature": 0.1, "max_length": 256}
)

# Define scoring prompt
scoring_prompt = PromptTemplate.from_template(
    """<|system|> Provide a single floating-point value between 0.0 and 1.0, representing the accuracy and completeness of the chatbot answer compared to the groundtruth. No explanations or additional text are necessary.
    <|user|>
Chatbot Answer:
{chatbot_answer}

Groundtruth Answer:
{groundtruth_answer}

Request:
Rate the chatbot answer between 0.0 and 1.0
<|assistant|>"""
)

# Define system message
system_message = """Your role as a chatbot assistant is to answer questions related to the University of the Philippines. \
    Politely decline if asked questions without relevant context."""

# Define question prompt
question_prompt = PromptTemplate.from_template(
    """[INST]
    {system_prompt}
Question:
{question}
[/INST]"""
)

# Define judge runnable
judge_runnable = scoring_prompt | llm_judge | StrOutputParser()

# Define chatbot runnable
cb_runnable = question_prompt | llm | StrOutputParser()

# Measure program duration
start_time = time.time()

# Process each row in the dataframe
answer_pairs = []

for index, row in df.iterrows():
    question = row['question ']
    gt = row['answer']

    # Invoke chatbot runnable
    cb_answer = cb_runnable.invoke({"system_prompt": system_message, "question": question})

    # Invoke judge runnable
    rating = judge_runnable.invoke({"chatbot_answer": cb_answer, "groundtruth_answer": gt})

    match = re.search(r"\d+\.\d+", rating)

    float_value = 0.0
    if match:
        float_value = float(match.group())
        print("Float value:", float_value)  # Output: Float value: 1.0
    else:
        print("No float value found in the text.")

    exactmatch_score, ngram_score = calculate_similarity_score(gt, cb_answer)

    # Append answer pair to the list
    answer_pairs.append({
        "question": question,
        "groundtruth": gt,
        "chatbot": cb_answer,
        "zephyr_score": float_value,
        "exactmatch_score": exactmatch_score,
        "ngram_score": ngram_score
    })

# Create dataframe from answer pairs
df_answer_pairs = pd.DataFrame(answer_pairs)

# Write dataframe to Excel file
df_answer_pairs.to_excel("mistral_rating.xlsx", index=False)

# Calculate program duration
duration = time.time() - start_time
print("Program duration:", duration, "seconds")
