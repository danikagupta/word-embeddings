# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import numpy as np

import streamlit as st
from streamlit.hello.utils import show_code
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns




def embedding_heatmap_demo() -> None:
    # Initialize the model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    txt1="The quick brown fox jumps over the lazy dog"
    txt2="Where there is smoke, there is fire."
    txt3="All good things must come to an end"
    txt4="An apple a day keeps the doctor away."
    txt5="Eat fruits daily to not see medical professional"

    txt1=st.text_input(value=txt1,label="Enter text 1")
    txt2=st.text_input(value=txt2,label="Enter text 2")
    txt3=st.text_input(value=txt3,label="Enter text 3")
    txt4=st.text_input(value=txt4,label="Enter text 4")
    txt5=st.text_input(value=txt5,label="Enter text 5")

    # Define text passages
    texts = [txt1,txt2,txt3,txt4,txt5]
    # Convert texts to embeddings
    embeddings = model.encode(texts)
    #print(embeddings)
    #print(embeddings.shape)

    # Calculate cosine similarity between each pair of texts
    similarity_matrix = cosine_similarity(embeddings)

    #Plot heat map
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', xticklabels=texts, yticklabels=texts)
    plt.title("Cosine Similarity Heatmap")
    #plt.show()
    st.pyplot(plt)


st.set_page_config(page_title="Embedding Heatmap Demo", page_icon="📹")
st.markdown("# Embedding heatmap Demo")
st.write(
    """This app shows how you can use Streamlit to build cool animations.
It displays an animated fractal based on the the Julia Set. Use the slider
to tune different parameters."""
)

embedding_heatmap_demo()

show_code(embedding_heatmap_demo)
