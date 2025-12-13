from transformers import pipeline

model_choice = input("Please enter the number of the model you wish to use: \n1. Sentiment Analysis\n2. Zero - shot Classification\n3. Text Generation\n4. Mask Filling\n5. Named Entity Recognition\n6. Queston Answering\n7. Text Summarization\n8. Text Translation\n9. Image Classification\n10. Automatic Speech Recognition\nYour Choice: ")

if model_choice == "1": 

    # Sentiment Analysis

    sentiment_classifier = pipeline("sentiment-analysis")
    sentiment_result = sentiment_classifier("I've loved HuggingFace courses my whole life.")

    print(f"Sentiment analysis result = {sentiment_result}")

elif model_choice == "2": 

    # Zero - shot classification

    zero_classifier = pipeline("zero-shot-classification")
    zero_result = zero_classifier("This is a course about the transformers library.", candidate_labels = ["educaton", "politics", "business"])

    print(f"Zero - shot classification result = {zero_result}")

elif model_choice == "3": 

    # Text generation

    pre_generated_text = "In this project it can be observed that"

    generator = pipeline("text-generation")
    generator_completed_text = generator(pre_generated_text, max_length = 50, num_return_sequences = 1)

    print(f"Pre - generated text = {pre_generated_text}")
    print(f"Text generation result = {generator_completed_text}")

elif model_choice == "4": 

    # Mask filling

    unmasker = pipeline("fill-mask")
    unmasker_result = unmasker("This is a course about the <mask> library.", top_k=2)

    print(f"Unmasker result = {unmasker_result}")

elif model_choice == "5":

    # Named Entity Recognition

    ner = pipeline("ner", grouped_entities = True)
    ner_result = ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")

    print(f"Named Entity Recognition result = {ner_result}")

elif model_choice == "6": 

    # Question Answering

    question_answerer = pipeline("question-answering")
    question_answerer_result = question_answerer(
        question="This project is being built for which class?",
        context=" This project is built for EE563 class for practicing LLM.",
    )

    print(f"Question Answerer result = {question_answerer_result}")

elif model_choice == "7":

    # Text Summarization

    text_to_be_summarized =     """
        America has changed dramatically during recent years. Not only has the number of 
        graduates in traditional engineering disciplines such as mechanical, civil, 
        electrical, chemical, and aeronautical engineering declined, but in most of 
        the premier American universities engineering curricula now concentrate on 
        and encourage largely the study of engineering science. As a result, there 
        are declining offerings in engineering subjects dealing with infrastructure, 
        the environment, and related issues, and greater concentration on high 
        technology subjects, largely supporting increasingly complex scientific 
        developments. While the latter is important, it should not be at the expense 
        of more traditional engineering.

        Rapidly developing economies such as China and India, as well as other 
        industrial countries in Europe and Asia, continue to encourage and advance 
        the teaching of engineering. Both China and India, respectively, graduate 
        six and eight times as many traditional engineers as does the United States. 
        Other industrial countries at minimum maintain their output, while America 
        suffers an increasingly serious decline in the number of engineering graduates 
        and a lack of well-educated engineers.
    """
    summarizer = pipeline("summarization")
    summarizer_result = summarizer(text_to_be_summarized)

    print(f"Before summarization = {text_to_be_summarized}")
    print(f"Summarizer result = {summarizer_result}")

elif model_choice == "8":

    #Text Translation

    text_to_be_translated = input("Enter an Turkish sentence to translate it to English: \n")
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-tr-en")
    translator_result = translator(text_to_be_translated)

    print(f"Translation result = {translator_result}")

elif model_choice == "9":

    # Image Classification

    image_classifier = pipeline(task="image-classification", model="google/vit-base-patch16-224")
    image_classifier_result = image_classifier("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
    print(image_classifier_result)

elif model_choice == "10":

    # Automatic Speech Recognition

    transcriber = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
    transcriber_result = transcriber("mlk.flac")
    print(transcriber_result)