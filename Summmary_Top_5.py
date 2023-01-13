#!/usr/bin/env python
# coding: utf-8

# In[3]:


content = "China’s Huawei overtook Samsung Electronics as the world’s biggest seller of mobile phones in the second quarter of 2020, shipping 55.8 million devices compared to Samsung’s 53.7 million, according to data from research firm Canalys. While Huawei’s sales fell 5 per cent from the same quarter a year earlier, South Korea’s Samsung posted a bigger drop of 30 per cent, owing to disruption from the coronavirus in key markets such as Brazil, the United States and Europe, Canalys said. Huawei’s overseas shipments fell 27 per cent in Q2 from a year earlier, but the company increased its dominance of the China market which has been faster to recover from COVID-19 and where it now sells over 70 per cent of its phones. “Our business has demonstrated exceptional resilience in these difficult times,” a Huawei spokesman said. “Amidst a period of unprecedented global economic slowdown and challenges, we’re continued to grow and further our leadership position.” Nevertheless, Huawei’s position as number one seller may prove short-lived once other markets recover given it is mainly due to economic disruption, a senior Huawei employee with knowledge of the matter told Reuters. Apple is due to release its Q2 iPhone shipment data on Friday."


# In[4]:


from transformers import T5Tokenizer, T5ForConditionalGeneration

T5_PATH = 't5-large' # T5 model name

# initialize the model architecture and weights

t5_model = T5ForConditionalGeneration.from_pretrained(T5_PATH)

# initialize the model tokenizer

t5_tokenizer = T5Tokenizer.from_pretrained(T5_PATH)


# In[ ]:


inputs = t5_tokenizer.encode("summarize: " + content, return_tensors="pt", max_length=512, padding='max_length', truncation=True)


# In[6]:


summary_ids = t5_model.generate(inputs,

                                    num_beams=int(2),

                                    no_repeat_ngram_size=3,

                                    length_penalty=2.0,

                                    min_length=min_length,

                                    max_length=max_length,

                                    early_stopping=True)

output = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

