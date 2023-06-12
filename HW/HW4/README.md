# HW4



### 呼叫 openai api 寫一個程式或系統



* 翻譯程式

> 程式碼

```python
import openai

def translate_text(text_to_translate, target_language):
    api_key = "your_api_key"  # 請替換為您的實際 API 金鑰
    openai.api_key = api_key

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=text_to_translate,
        max_tokens=100,
        temperature=0.7,
        n=1,
        stop=None,
        temperature=0.7
    )

    translated_text = response.choices[0].text.strip()

    return translated_text

# 使用者輸入要翻譯的內容和目標語言
text_to_translate = input("請輸入要翻譯的內容：")
target_language = input("請輸入要翻譯成的語言：")
translated_text = translate_text(text_to_translate, target_language)
print("翻譯結果：", translated_text)
```



> 來源

* ChatGPT (全由chatgpt生成 無更動)

![](D:\大學\大三\大三下\人工智慧\HW\HW4\HW4.jpg)