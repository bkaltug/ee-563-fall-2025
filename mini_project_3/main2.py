import gradio as gr
from transformers import pipeline


class AITaskHandler:
   
    def __init__(self):
        self.pipelines = {}

    def _get_pipeline(self, task, model=None, grouped_entities=False):
        key = f"{task}_{model}"
        if key not in self.pipelines:
            print(f"Loading model for {task}...") 
            if grouped_entities:
                 self.pipelines[key] = pipeline(task, model=model, grouped_entities=True)
            else:
                 self.pipelines[key] = pipeline(task, model=model) if model else pipeline(task)
        return self.pipelines[key]

    def sentiment_analysis(self, text):
        classifier = self._get_pipeline("sentiment-analysis")
        result = classifier(text)
        return f"Label: {result[0]['label']}, Score: {result[0]['score']:.4f}"

    def zero_shot_classification(self, text, labels_text):
        classifier = self._get_pipeline("zero-shot-classification")
        candidate_labels = [label.strip() for label in labels_text.split(",")]
        result = classifier(text, candidate_labels=candidate_labels)
        return result

    def text_generation(self, text):
        generator = self._get_pipeline("text-generation")
        result = generator(text, max_length=50, num_return_sequences=0)
        return result[0]['generated_text']

    def mask_filling(self, text):
        unmasker = self._get_pipeline("fill-mask")
        result = unmasker(text, top_k=2)
        return "\n".join([f"{res['token_str']} (Score: {res['score']:.4f})" for res in result])

    def named_entity_recognition(self, text):
        ner = self._get_pipeline("ner", grouped_entities=True)
        result = ner(text)
        return result

    def question_answering(self, context, question):
        qa_model = self._get_pipeline("question-answering")
        result = qa_model(question=question, context=context)
        return f"Answer: {result['answer']} (Score: {result['score']:.4f})"

    def text_summarization(self, text):
        summarizer = self._get_pipeline("summarization")
        result = summarizer(text)
        return result[0]['summary_text']

    def text_translation(self, text):
        translator = self._get_pipeline("translation", model="Helsinki-NLP/opus-mt-tr-en")
        result = translator(text)
        return result[0]['translation_text']

    def image_classification(self, image):
        if image is None:
            return "Please upload an image."
        classifier = self._get_pipeline("image-classification", model="google/vit-base-patch16-224")
        result = classifier(image)
        return {res['label']: res['score'] for res in result}

    def automatic_speech_recognition(self, audio):
        if audio is None:
            return "Please upload or record audio."
        transcriber = self._get_pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")
        result = transcriber(audio)
        return result['text']


handler = AITaskHandler()


with gr.Blocks(title="EE563 AI Project") as demo:
    gr.Markdown("# EE 563 Mini Project #3 - Berkay Altuğ Ustagül")

    with gr.Tab("1. Sentiment Analysis"):
        s_input = gr.Textbox(label="Input Text", value="I've loved HuggingFace courses my whole life.")
        s_button = gr.Button("Analyze Sentiment")
        s_output = gr.Textbox(label="Result")
        s_button.click(handler.sentiment_analysis, inputs=s_input, outputs=s_output)

    with gr.Tab("2. Zero-Shot Classification"):
        z_input = gr.Textbox(label="Input Text", value="This is a course about the transformers library.")
        z_labels = gr.Textbox(label="Candidate Labels (comma separated)", value="education, politics, business")
        z_button = gr.Button("Classify")
        z_output = gr.JSON(label="Result")
        z_button.click(handler.zero_shot_classification, inputs=[z_input, z_labels], outputs=z_output)

    with gr.Tab("3. Text Generation"):
        g_input = gr.Textbox(label="Start of Sentence", value="In this project it can be observed that")
        g_button = gr.Button("Generate Text")
        g_output = gr.Textbox(label="Completed Text", lines=10)
        g_button.click(handler.text_generation, inputs=g_input, outputs=g_output)

    with gr.Tab("4. Mask Filling"):
        m_input = gr.Textbox(label="Input Text (use <mask>)", value="This is a course about the <mask> library.")
        m_button = gr.Button("Fill Mask")
        m_output = gr.Textbox(label="Predictions", lines = 3)
        m_button.click(handler.mask_filling, inputs=m_input, outputs=m_output)

    with gr.Tab("5. NER"):
        n_input = gr.Textbox(label="Input Text", value="My name is Sylvain and I work at Hugging Face in Brooklyn.")
        n_button = gr.Button("Identify Entities")
        n_output = gr.JSON(label="Entities")
        n_button.click(handler.named_entity_recognition, inputs=n_input, outputs=n_output)

    with gr.Tab("6. Question Answering"):
        q_context = gr.Textbox(label="Context", lines=4, value="This project is built for EE563 class for practicing LLM.")
        q_question = gr.Textbox(label="Question", value="This project is being built for which class?")
        q_button = gr.Button("Answer")
        q_output = gr.Textbox(label="Answer")
        q_button.click(handler.question_answering, inputs=[q_context, q_question], outputs=q_output)

    with gr.Tab("7. Summarization"):
        sum_text = """America has changed dramatically during recent years. Not only has the number of 
        graduates in traditional engineering disciplines such as mechanical, civil, 
        electrical, chemical, and aeronautical engineering declined, but in most of 
        the premier American universities engineering curricula now concentrate on 
        and encourage largely the study of engineering science."""
        
        sum_input = gr.Textbox(label="Long Text", lines=5, value=sum_text)
        sum_button = gr.Button("Summarize")
        sum_output = gr.Textbox(label="Summary", lines=10)
        sum_button.click(handler.text_summarization, inputs=sum_input, outputs=sum_output)

    with gr.Tab("8. Translation"):
        t_input = gr.Textbox(label="Turkish Text", placeholder="Enter Turkish sentence...")
        t_button = gr.Button("Translate")
        t_output = gr.Textbox(label="English Translation")
        t_button.click(handler.text_translation, inputs=t_input, outputs=t_output)

    with gr.Tab("9. Image Classification"):
        i_input = gr.Image(type="pil", label="Upload Image")
        i_button = gr.Button("Classify Image")
        i_output = gr.Label(num_top_classes=3)
        i_button.click(handler.image_classification, inputs=i_input, outputs=i_output)

    with gr.Tab("10. Audio Speech Recognition"):
        a_input = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio File")
        a_button = gr.Button("Transcribe")
        a_output = gr.Textbox(label="Transcription")
        a_button.click(handler.automatic_speech_recognition, inputs=a_input, outputs=a_output)

if __name__ == "__main__":
    demo.launch()