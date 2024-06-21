### Repository Name: Spanglish LLM Fine-tuned with Cohere AI

#### Overview
This repository contains code and resources to fine-tune a Large Language Model (LLM) using Cohere AI for the purpose of generating text in Spanglish, a hybrid language mixing Spanish and English. The fine-tuning process leverages Cohere AI's capabilities to adapt the LLM to the linguistic nuances and structures of Spanglish.

#### Dependencies
- Python 3.x
- Cohere AI Python SDK
- Transformers library (Hugging Face)
- Other dependencies as listed in `requirements.txt`

#### Getting Started
1. **Installation**
   ```bash
   pip install -r requirements.txt
   ```

2. **Fine-tuning the Model**
   - Prepare your dataset containing Spanglish text.
   - Use Cohere AI's platform to fine-tune the LLM on your dataset. Refer to Cohere AI's documentation for detailed instructions.

3. **Running the Model**
   - Once fine-tuned, use the provided scripts to generate Spanglish text or integrate the model into your applications.

#### Folder Structure
- **cohere/**: Contains data for exapmle Spanglish conversations used for finetuning.
- **new_plots/**: Has plots relating to gerryfair fairness tests.

#### GerryFair Fairness Algorithm
This repository implements the GerryFair fairness algorithm to address ethical considerations related to Large Language Models (LLMs), particularly in the context of Spanglish generation. The GerryFair algorithm helps mitigate biases and ensures that the generated text respects linguistic diversity and fairness principles. 

#### License
This repository is licensed under the MIT License. See `LICENSE` for more information.

#### Contributors
- Sarthak Agrawal (https://github.com/sarthyparty)
- Vivek Saravanan (https://github.com/viveks295)

#### Acknowledgments
- Cohere AI for providing the fine-tuning platform and support.
- Hugging Face for the Transformers library.
- The GerryFair algorithm contributors for advancing fairness in LLMs. (https://arxiv.org/abs/1711.05144)

#### Issues
If you encounter any issues or have suggestions, please open an issue on GitHub.

#### Disclaimer
This project is for research and educational purposes. Consider the ethical implications of language model usage, especially in multilingual contexts like Spanglish.

#### Contact
For questions or collaborations, contact vsaravanan@wisc.edu
