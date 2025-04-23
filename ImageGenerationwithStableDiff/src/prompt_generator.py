import random

class PromptGenerator:
    '''
    This class creates random prompts for image generation.
    Following methods implemented for generating images similar to FBA dataset (contains 4 categories): generate_with_one_cat, generate_random_prompt_with_a_category, generate_random_prompt
    Following method implemented for generating images similar to Tool dataset (contains single category): generate_random_prompt_for_tool_dataset
    '''
    @staticmethod
    def generate_with_one_cat(tool_type, cat):
        '''
        Generates a prompt with one category.
        Args:
            tool_type (str): The type of tool.
            cat (str): The category to include in the prompt.
        Returns: prompt(str)
        '''
        return f"{tool_type} with {cat} wear"
    
    @staticmethod
    def generate_random_prompt_with_a_category(tool_type, categories, fixed_cat):
        '''
        Generates a prompt with a fixed category and random categories.
        Args:
            tool_type (str): The type of tool.
            categories (list): List of categories to choose from.
            fixed_cat (str): The category to include in the prompt.
        Returns: prompt(str)
        '''
        if fixed_cat not in categories:
            raise ValueError("Invalid category provided.")
        
        temp_categories = categories.copy()
        temp_categories.remove(fixed_cat)
        num_random_categories = random.randint(0, len(temp_categories))
        random_categories = random.sample(temp_categories, num_random_categories)

        # Create a list with all selected categories
        all_categories = random_categories + [fixed_cat]
        # Shuffle the list
        random.shuffle(all_categories)

        prompt = f"{tool_type} with {', '.join(all_categories)} wear"
        return prompt
    
    @staticmethod
    def generate_random_prompt(tool_type, categories):
        '''
        Generates a prompt with random categories.
        Args:
            tool_type (str): The type of tool.
            categories (list): List of categories to choose from.
        Returns: prompt(str)
        '''
        temp_categories = categories.copy()
        
        # select at least one category
        num_random_categories = random.randint(1, len(temp_categories))

        # select categories randomly
        random_categories = random.sample(temp_categories, num_random_categories)

        prompt = f"{tool_type} with {', '.join(random_categories)} wear"
        return prompt
    
    @staticmethod
    def generate_random_prompt_for_tool_dataset(tool_type, wear_severities):
        '''
        Tool dataset does not contain wear categories instead it contains wear severity as low, moderate and high.
        Generates a prompt for the tool dataset with random wear severity.
        Args:
            tool_type (str): The type of tool.
            wear_severities (list): List of wear severities to choose from.
        Returns: prompt(str)
        '''
        random_severity = random.choice(wear_severities)

        prompt = f"{tool_type} with {random_severity}"
        return prompt

    



