import openai
def askChatGPT(question):
    openai.api_key = "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"
    prompt = question
    model_engine = "text-davinci-003"

    completions = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    message = completions.choices[0].text
    return(message)
if __name__ == "__main__":
    askChatGPT("give the key nouns in the sentence that depict an entity in the scene. Only entities that will appear in the scene, not other nouns, should be given. Your response should only contain a list of nouns separated by commas. [Example: Sentence:'a blurry image of two people standing next to each other in a dark room'. Response: people, room] Now we have the Sentence: 'High-quality photo of a rugged man enjoying a smoke break by the sea, capturing the sense of freedom and relaxation. Shot in high definition, this image is perfect for any project that requires a cool and laid-back vibe. A man wearing a black jacket and jeans, with a black hat and sunglasses, holding a cigarette in his right hand and his left hand in his pocket, standing on the beach with blue skies and sea waves in the background. ' ")