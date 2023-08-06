# Dialog Summarization with Generative AI using LLM (Large Language Models)
* Lange Language model used here is FLAN-T2 on Huggingface
* First notebook, performed prompt engineering is performed towards the task needed to achive by mentioning respective prompt in plain english. Compared zero shot, one shot, and few shot inferences. This give an idea about on how there is scope to enhance the generative output of Large Language Models.
* Second notebook, performed the full fine tuning, PEFT (Parameter Efficient Fine Tuning) is perfomed using LoRA (Low Rank Adaption) method. PEFT provides the finetuning with less computational resources as compared to original LLM, and gives cost benefit.
* Summarizing the learning as below

### Summary of my learnings after performing below fine tuning exercises
* Got the fair idea of how PEFT'ed models gives performance very close to fully trained LLM giving the cost benefit of lesser resources required for training
* Learnings related to fully fine tuned LLM and PEFT LLM model preparation for inference shared below
#### Full finetuned LLM model preparation for inference
* For full fine tuning of the model, create a prompt with prepent the dialog by instruction at the begining as "Summarize the conversation" and append the "Summary:" to the end of the dialogue. 
* Consider these promps as x_trains, and y_trains as original labels i.e. "summary" in the original dataset. <br><br>
* start_prompt = 'Summarize the following conversation.\n\n'
* end_prompt = '\n\nSummary: '
* prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dialogue"]]
* example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
* example['labels'] = tokenizer(example["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids

#### PEFT LLM model preparation for inference
* For PEFT, here LoRA (Low rank adaption) is used. 
* Get the lora adapter using the original model first with  "peft_model = get_peft_model(original_model, lora_config)" by passing the lora_configs. 
* Then, train the PEFT adapter using "peft_trainer = Trainer(model=peft_model,args= peft_training_args, train_dataset=tokenized_datasets["train"])
* Train the peft_trainer and save the adapter model to './peft-dialogue-summary-checkpoint-from-s3/' . 
* Prepare this adapter model by adding it with original FLAN-T5 model as model_peft_adapter_combined_with_original
* “model_peft_adapter_combined_with_original = PeftModel.from_pretrained(original_model_base, './peft-dialogue-summary-checkpoint-from-s3/', torch_dtype=torch.bfloat16, is_trainable=False)”
* Then use the model_peft_adapter_combined_with_original model for evaluation.

#### ROUGE metrics
* ROUGE performance metric measures or evaluates the performance of predictions as compared to the original labels in case of text
* when my label itself is text, in that case ROUGE metrics can be used as performance metric to evaluate that supervised learning model.

Citations & Credits - These are the hands-on execise lab assignment completed as part of "Generative AI with Large Language Models" couse on Coursera.com.
