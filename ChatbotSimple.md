**# streamlit run ChatbotSimple.py**
    import streamlit as st
    import os
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.schema import HumanMessage, AIMessage
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationChain

# Set Google API key

    GOOGLE_API_KEY="XXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    
    st.title("Chat with Gemini Model")

# Initialize LangChain Gemini model and memory
    if "chain" not in st.session_state and GOOGLE_API_KEY:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)
        memory = ConversationBufferMemory(return_messages=True)
        st.session_state.chain = ConversationChain(llm=llm, memory=memory, verbose=False)

# Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

# Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
    if prompt := st.chat_input("Say something..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
    
        # Get response from LangChain chain
        response = st.session_state.chain.run(prompt)
        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    

<img width="1082" alt="image" src="https://github.com/user-attachments/assets/18210b6b-63ab-4abe-ab4e-841424b1640f" />

