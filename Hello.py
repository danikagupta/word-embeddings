import streamlit as st

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import inspect
import textwrap

st.set_page_config(layout="wide")

def embedding_heatmap_demo() -> None:
    # Initialize the model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    txt1="The quick brown fox jumps over the lazy dog."
    txt2="Where there is smoke, there is fire."
    txt3="An apple a day keeps the doctor away."
    txt4="Eat fruits daily to not see medical professional."

    st.markdown("# Provide up to four text segments to compare")
    col1,col2,col3,col4 = st.columns(4)
    txt1=col1.text_area(value=txt1,label="Text 1",label_visibility="collapsed")
    txt2=col2.text_area(value=txt2,label="Text 2",label_visibility="collapsed")
    txt3=col3.text_area(value=txt3,label="Text 3",label_visibility="collapsed")
    txt4=col4.text_area(value=txt4,label="Text 4",label_visibility="collapsed")

    texts = [txt1,txt2,txt3,txt4]
    embeddings = model.encode(texts)
    similarity_matrix = cosine_similarity(embeddings)

    #Plot heat map
    sns.set_context('talk')  
    plt.figure(figsize=(8, 6))  
    truncated_texts = [text[:20] for text in texts]
    sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', xticklabels=truncated_texts, 
                yticklabels=truncated_texts,annot_kws={"size": 18, "weight": "bold"})
    plt.xticks(rotation=45, ha='right', fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("Cosine Similarity Heatmap")
    #plt.show()
    st.pyplot(plt)

    # Show embeddings
    df = pd.DataFrame(embeddings)
    df.insert(0, "Text", texts)
    with st.expander("## See embeddings"):
        st.dataframe(df,hide_index=True)

    # Show code
    with st.expander("## See app code"):
        sourcelines, _ = inspect.getsourcelines(embedding_heatmap_demo)
        st.code(textwrap.dedent("".join(sourcelines[1:])))

embedding_heatmap_demo()