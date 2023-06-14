import openai

openai.api_key = 'YOUR_API_KEY'

def translate_text(text, target_language):
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=f'Translate the following text from {target_language} to English:\n{text}\n--\nTranslate to: en',
        temperature=0,
        max_tokens=128,
        n=1,
        stop=None
    )

    translation = response.choices[0].text.strip()
    return translation

input_text = input("請輸入要翻譯的文字：")
target_language = input("請輸入目標語言（例如：en表示英文，zh-TW表示中文繁體）：")

translated_text = translate_text(input_text, target_language)
print("翻譯結果：", translated_text)
