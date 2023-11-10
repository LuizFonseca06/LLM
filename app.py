# Import os to set API key
import os
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Bring in streamlit for UI/app interface
import streamlit as st
from PIL import Image
# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)
# Import PDF document loaders...there's other ones as well!
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
# Import OpenAI as main LLM service
from langchain.llms import OpenAI
# Import chroma as the vector store
from langchain.vectorstores import Chroma

from config import OPENAI_API_KEY


# FAQ:
# what is the purpose of this manual?
# what comes inside the box?
# when should the battery be changed?


def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmpfile:
            # Write the contents of the uploaded file to the temp file
            tmpfile.write(uploaded_file.read())
            return tmpfile.name  # Return the path to the temp file
    except Exception as e:
        # If there's an error, print it out on the Streamlit app
        st.error(f'Error saving file: {e}')
        return None





def main():
    st.title('🦜🔗 PDF-Chat: Interact with Your PDFs in a Conversational Way')
    st.subheader('Load your PDF, ask questions, and receive answers directly from the document.')

    image = Image.open('images/image_title.png')
    st.image(image)

    file_path = os.getcwd()

    uploaded_file = st.file_uploader("Choose a PDF file", type='pdf')
    success = False
    if uploaded_file is not None:
        file_path = save_uploaded_file(uploaded_file)
        if file_path is not None:
            st.success(f'Success! PDF has been saved to: {file_path}')
            success = True
        else:
            st.error('An error occurred while saving the file.')
    if success:
        loader = PyPDFLoader(file_path)

        # chunks = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,  # characters
            chunk_overlap=50,  # overlap between chunks in characters
            length_function=len,  # function to determine the length of a chunk
        )
        chunks = loader.load_and_split(text_splitter=text_splitter)

        # Create instance of OpenAI LLM
        llm = OpenAI(
            temperature=0.1,
            verbose=True,
            openai_api_key=OPENAI_API_KEY
        )
        embeddings = OpenAIEmbeddings()

        store = Chroma.from_documents(chunks, embeddings, collection_name='pdf_collection')

        retriever = store.as_retriever()
        # Lanhchain Stuff
        vectorstore_info = VectorStoreInfo(
            name="pdf_collection",
            description=" A pdf file to answer your questions",
            vectorstore=store
        )
        toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)
        agent_executor = create_vectorstore_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True
        )

    # Create a text input box for the user
    prompt = st.text_input('Input your prompt here')

    # If the user hits enter
    if prompt:
        # Then pass the prompt to the LLM
        response = agent_executor.run(prompt)
        #relevant_docs = retriever.get_relevant_documents(prompt)

        # ...and write it out to the screen
        st.write(response)

        # With a streamlit expander
        with st.expander('Document Similarity Search'):
            # Find the relevant pages
            search = store.similarity_search_with_score(prompt)
            # Write out the first
            st.write(search[0][0].page_content)


if __name__ == "__main__":
    main()
