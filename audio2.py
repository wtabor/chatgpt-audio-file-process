import gradio as gr
import openai, config

openai.api_key = config.OPENAI_API_KEY

# Define initial system message
messages = [{"role": "system", "content": 'Can you summarize these meeting notes?.'}]

# Define available prompts for user to choose from
prompts = [
    "Introduce yourself and tell me what you do.",
    "What are your top three priorities for the week?",
    "Can you summarize the meeting notes?",
    "What are your thoughts on the latest market trends?",
]

def transcribe(prompt, audio):
    global messages

    audio_file = open(audio, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)

    messages.append({"role": "user", "content": transcript["text"]})

    # Call OpenAI API only if there is a user message
    if messages[-1]["role"] == "user":
        # Use "Summarize meeting notes" prompt if selected
        if prompt == "Can you summarize the meeting notes?":
            prompt_text = "Summarize the following meeting notes:"
        else:
            prompt_text = prompt

        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt='\n'.join([f'{m["role"]}: {m["content"]}' for m in messages] + [prompt_text]),
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.7,
        )

        system_message = response.choices[0].text.strip()
        messages.append({"role": "system", "content": system_message})

    chat_transcript = ""
    for message in messages:
        chat_transcript += message["role"] + ": " + message["content"] + "\n\n"

    return chat_transcript

# Set up Gradio interface
ui = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Dropdown(prompts, label="Choose a prompt:"),
        gr.Audio(source="upload", type="filepath", label="Upload your audio:")
    ],
    outputs="text",
    title="Investment Banker Personal Assistant",
    description="This app provides a personal assistant for an investment banker that can transcribe voice input and respond to prompts using OpenAI's powerful language model."
)

# Launch Gradio interface
ui.launch()
