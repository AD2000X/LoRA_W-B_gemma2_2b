{
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Loading Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Download Dolly Dataset\n",
    "!wget -O data/databricks-dolly-15k.jsonl https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Loading Dataset\n",
    "data = load_dolly_dataset(max_samples=1000)\n",
    "\n",
    "# Display Example\n",
    "print(\"Sample Materials:\")\n",
    "print(\"-\" * 50)\n",
    "print(data[0])\n",
    "print(\"-\" * 50)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Setting up the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Set up the model with FP32 precision and enable LoRA\n",
    "gemma_lm = setup_model(precision=\"fp32\", lora_rank=4)\n",
    "gemma_lm.summary()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Original model reasoning\n",
    "\n",
    "In this section, we will query the model with various prompts and see how it responds.。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def generate_text(prompt, model):\n",
    "    \"\"\"Generating text\"\"\"\n",
    "    template = \"Instruction:\\n{}\\n\\nResponse:\\n{}\"\n",
    "    full_prompt = template.format(prompt, \"\")\n",
    "    \n",
    "    sampler = keras_nlp.samplers.TopKSampler(k=5, seed=42)\n",
    "    model.compile(sampler=sampler)\n",
    "    \n",
    "    return model.generate(full_prompt, max_length=256)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Europe Travel Tips\n",
    "print(\"Original model response:\")\n",
    "print(generate_text(\"What should I do on a trip to Europe?\", gemma_lm))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Photosynthesis Explanation Tips\n",
    "print(\"Original model response:\")\n",
    "print(generate_text(\"Explain the process of photosynthesis in a way that a child could understand.\", gemma_lm))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Training the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Set callback function\n",
    "callbacks = get_callbacks()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Compile the model\n",
    "gemma_lm = compile_model(\n",
    "    gemma_lm,\n",
    "    learning_rate=5e-5,\n",
    "    weight_decay=0.01\n",
    ")\n",
    "\n",
    "# Training the model\n",
    "start_time = time.time()\n",
    "history, gemma_lm = train_model(\n",
    "    gemma_lm,\n",
    "    data,\n",
    "    epochs=20,\n",
    "    batch_size=4,\n",
    "    callbacks=callbacks\n",
    ")\n",
    "training_time = time.time() - start_time\n",
    "\n",
    "print(f\"Training completed, time consuming {training_time:.2f} Seconds\")\n",
    "\n",
    "# Record training time\n",
    "wandb.log({\"training_time_seconds\": training_time})"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Inference after fine-tuning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Europe Travel Tips\n",
    "europe_response = generate_text(\"What should I do on a trip to Europe?\", gemma_lm)\n",
    "print(\"Model response after fine-tuning:\")\n",
    "print(\"-\" * 50)\n",
    "print(europe_response)\n",
    "print(\"-\" * 50)\n",
    "\n",
    "# Evaluating text quality\n",
    "evaluator = TextEvaluator()\n",
    "europe_text = europe_response.split(\"Response:\\n\")[1].strip()\n",
    "results = evaluator.evaluate_text(europe_text)\n",
    "print(\"\\nEvaluation results:\")\n",
    "for metric, value in results.items():\n",
    "    print(f\"{metric}: {value}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Photosynthesis Explanation Tips\n",
    "photo_response = generate_text(\"Explain the process of photosynthesis in a way that a child could understand.\", gemma_lm)\n",
    "print(\"Model response after fine-tuning:\")\n",
    "print(\"-\" * 50)\n",
    "print(photo_response)\n",
    "print(\"-\" * 50)\n",
    "\n",
    "# Evaluating text quality\n",
    "photo_text = photo_response.split(\"Response:\\n\")[1].strip()\n",
    "results = evaluator.evaluate_text(photo_text)\n",
    "print(\"\\nEvaluation results:\")\n",
    "for metric, value in results.items():\n",
    "    print(f\"{metric}: {value}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Save the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save the model\n",
    "save_path = save_model(gemma_lm, \"../models/gemma2-lora-fp32\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Complete W&B experiment\n",
    "wandb.finish()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}