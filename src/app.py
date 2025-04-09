import streamlit as st
from src.utils.qa_handler import QAHandler

def main():
    st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")
    
    # Initialize session state
    if "qa_handler" not in st.session_state:
        st.session_state.qa_handler = QAHandler()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_model" not in st.session_state:
        st.session_state.current_model = "llama2"

    # Sidebar for model selection
    with st.sidebar:
        st.title("Settings")
        
        # Refresh button for models
        if st.button("🔄 Refresh Models"):
            st.session_state.qa_handler = QAHandler()  # Reinitialize to refresh model list
        
        # Get available models
        available_models = st.session_state.qa_handler.get_available_models()
        
        # Model selection
        selected_model = st.selectbox(
            "Select Model",
            available_models,
            index=available_models.index(st.session_state.current_model) if st.session_state.current_model in available_models else 0
        )
        
        # Update model if changed
        if selected_model != st.session_state.current_model:
            if st.session_state.qa_handler.set_model(selected_model):
                st.session_state.current_model = selected_model
                st.success(f"Model changed to {selected_model}")
            else:
                st.error(f"Failed to switch to model {selected_model}")
        
        # Document upload
        st.title("Document Upload")
        uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])
        
        if uploaded_file is not None:
            # Save the uploaded file temporarily
            with open("temp_upload", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process the document
            with st.spinner("Processing document..."):
                result = st.session_state.qa_handler.process_document("temp_upload")
                if result.startswith("Error"):
                    st.error(result)
                else:
                    st.success("Document processed successfully!")
                    st.info(f"Document contains {len(result.split())} words")

    # Main chat interface
    st.title("AI Assistant 🤖")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.qa_handler.get_response(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main() 