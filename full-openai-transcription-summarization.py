import gradio as gr
import openai
import os

openai.api_key = os.environ['OPENAI_API_KEY']

# Define initial system message
messages = [{"role": "system", "content":''}]

# Define available prompts for user to choose from
prompts = [
    "1. Create a detailed summary outline of the transcription in meeting notes organized by discussion topics and 2. List action items and key metrics or numbers discussed; use appropriate formatting for all notes such as bullet points, numbered lists, bold or italicized text to enhance the organization and readability of the notes.",
    "Can you summarize the meeting notes?",
]

def transcribe(prompt, audio):
    global messages

    audio_file = open(audio, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)

    messages.append({"role": "user", "content": transcript["text"]})

    # Call OpenAI API only if there is a user message
    if messages[-1]["role"] == "user":
        # Use selected prompt as the instructions for OpenAI
        prompt_text = prompt

        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt='\n'.join([m["content"] for m in messages] + [prompt_text]),
            max_tokens=3000,
            n=1,
            stop=None,
            temperature=0.7,
        )

        system_message = response.choices[0].text.strip()
        messages.append({"role": "system", "content": system_message})

        chat_transcript = ""
    for message in messages:
        if message["role"] == "system":
            chat_transcript += message["content"] + "\n\n"
        else:
            chat_transcript += message["content"] + "\n\n"

    return chat_transcript


    """ chat_transcript = ""
    for message in messages:
        chat_transcript += message["role"] + ": " + message["content"] + "\n\n"

    return chat_transcript """

# Set up Gradio interface
ui = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Dropdown(choices=prompts, label="Choose a prompt:", default=[0]),
        gr.Audio(source="upload", type="filepath", label="Upload your audio:")
    ],
    outputs="text",
    title="Investment Banker Personal Assistant",
    description="This app provides a personal assistant for an investment banker that can transcribe voice input and respond to prompts using OpenAI's powerful language model."
)

# Launch Gradio interface
ui.launch()
