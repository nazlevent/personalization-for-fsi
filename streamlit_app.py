import streamlit as st
import time
import vertexai
import base64
from vertexai.preview.vision_models import ImageGenerationModel
from vertexai.generative_models import GenerativeModel, Part, SafetySetting, FinishReason, FunctionDeclaration, Tool, Content
import vertexai.preview.generative_models as generative_models

YOUR_PROJECT_ID = "south-emea-ml-tools"
vertexai.init(project=YOUR_PROJECT_ID, location="us-central1")

#st.title(f"ING Demo")

st.markdown(
    """
    <style>
        .logo-text {
            font-weight: bold; /* Make text thicker */
            font-size: 24px;  /* Adjust the font size */
        }

        .logo-image {
            height: 60px; /* Adjust the logo height as needed */
            width: auto;  /* Maintain aspect ratio */
            margin-left: 10px; /* Add spacing between logo and text */
            vertical-align: middle; /* Vertically center the logo */
        }
    </style>

    <div style="display: flex; align-items: center;">
        <img class="logo-image" src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTzTY6KtRF4YiS9M6VIg7fljLqppoPbJ-mrLw&s" alt="Logo">
         <span class="logo-text">      </span>
        
       
    </div>
    """,
    unsafe_allow_html=True,
)

st.title(f"Personalization for FSI")

st.markdown(
    "This demo is created for the personalization in FSI."
)

def cancel_subscriptions(subscription_name):
    """
    mock API.
    """
    return "Done! I have sent an e-mail to cancel your " + subscription_name.capitalize() + " subscription."
    

# Define your function declarations
cancel_subscription_func = FunctionDeclaration(
    name="cancel_subscription",
    description="Cancel a user's subscription given name",
    parameters={
        "type": "object",
        "properties": {
            "subscription_name": {
                "type": "string",
                "description": "The name of the subscription to cancel",
            }
        },
    },
)

# Create a tool with your function
advisor_tool = Tool(function_declarations=[cancel_subscription_func])

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

safety_settings = [
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    ),
]

textsi_1 = "You are a financial advisor for a bank especially for gen Z customers."



model = GenerativeModel(
    "gemini-1.5-flash-001",
    system_instruction=[textsi_1]
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "multi_modal_messages" not in st.session_state:
    st.session_state.multi_modal_messages = []

tabs = st.tabs(["Chat","Multi-Modality","Image Generation"])



with tabs[0]:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        contentHistory = [
            Content(role=(m['role'] if m['role'] != 'assistant' else 'model'), parts=[Part.from_text(m['content'])])
            for m in st.session_state.messages[:-1]
        ]

        with st.chat_message("assistant"):
            chat = model.start_chat(
                history=contentHistory
            )
            last_message = st.session_state.messages[-1]
            responses = chat.send_message(last_message["content"],stream=True)
            text_response = []
            for chunk in responses:
                text_response.append(chunk.text if hasattr(chunk, "text") and chunk.text else "")
            response = st.write_stream(text_response)
            print(response)
        st.session_state.messages.append({"role": "assistant", "content": response})



with tabs[1]:
    
    #2 inputs and 1 button. button named "send", inputs are file upload and text input
    uploaded_file = st.file_uploader("Upload a file", type=["pdf"])
    text_input = st.text_input("Speak to your advisor :")
    send_button = st.button("Send")

    if send_button:
        if uploaded_file is not None and text_input:
            with st.spinner("Analyzing..."):
                document1 = Part.from_data(
                    mime_type="application/pdf",
                    data=base64.b64decode(base64.b64encode(uploaded_file.read()).decode("utf-8"))
                )
                response = model.generate_content(
                    [document1, text_input],
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    stream=False,
                )
                
                with st.chat_message("assistant"):
                    st.markdown(response.text)
        elif text_input:
            with st.spinner("Thinking..."):
                response = model.generate_content(
                    [text_input],
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    stream=False,
                    tools=[advisor_tool],
                )

                if (response.candidates[0].content.parts[0].function_call.name == "cancel_subscription"):
                    subscription_name_parameter = (
                        response.candidates[0].content.parts[0].function_call.args["subscription_name"]
                    )
                    response_text = cancel_subscriptions(subscription_name=subscription_name_parameter)
                    print(response_text)
                    with st.chat_message("assistant"):
                        st.markdown(response_text)
                else:
                    with st.chat_message("assistant"):
                        st.markdown(response.text)

# Image Generation Tab
with tabs[2]:
    image_prompt = st.text_input("Enter your image prompt:")
    generate_button = st.button("Generate")

    if generate_button:
        # Google ImageGen integration
        with st.spinner("Generating image..."):
            generation_model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")

            response = generation_model.generate_images(
                prompt= image_prompt,
                number_of_images=2,
                aspect_ratio="3:4",
            )
            
            count = 0
            image_files = []
            for image in response.images:
                count = count + 1
                file_name = "generated_image_" + str(count) + ".png"
                image.save(location=file_name, include_generation_parameters = False)
                image_files.append(file_name)
            
            st.image(image_files)

