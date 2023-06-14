# 期中作業 : 語言處理



### 原理

1. 中翻英：
   - 分詞：中文句子需要進行分詞，將句子劃分為一個個詞語或片語。
   - 語言模型：根據中文語言模型，給出每個詞語在該句子中的可能性，以確定最可能的詞語組合方式。
   - 語義分析：通過分析詞語之間的語義關係，了解句子的含義和語境。
   - 翻譯規則：利用事先設定的翻譯規則，將中文詞語轉換為對應的英文詞語或短語。
   - 生成翻譯：根據翻譯規則和語義分析的結果，生成最終的英文翻譯句子。
2. 英翻中：
   - 分詞：英文句子需要進行分詞，將句子劃分為一個個單詞或短語。
   - 語言模型：根據英文語言模型，給出每個單詞在該句子中的可能性，以確定最可能的單詞組合方式。
   - 語義分析：通過分析單詞之間的語義關係，了解句子的含義和語境。
   - 翻譯規則：利用事先設定的翻譯規則，將英文單詞或短語轉換為對應的中文詞語。
   - 生成翻譯：根據翻譯規則和語義分析的結果，生成最終的中文翻譯句子。



> 程式碼

```python
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
```



### 結果

![](https://github.com/Roy-Roo/ai111b/blob/main/HW/MidHW/MidHW.jpg)





### 來源

* ChatGPT (全由 ChatGPT 生成 ， 未修改)
