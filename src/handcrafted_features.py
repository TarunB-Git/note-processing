"""
Handcrafted feature functions for text classification.
This module contains functions to extract specific features from text notes.
"""

def get_question_vocab():
    """
    Returns a set of typical vocabulary words commonly found in questions.
    Includes WH-words and other question-related terms.
    """
    question_words = {
        # WH-words
        'what', 'why', 'when', 'where', 'how', 'which', 'who', 'whom',
        # Other question indicators
        'identify', 'reason', 'explain', 'describe', 'define', 'list',
        'name', 'tell', 'show', 'find', 'determine', 'calculate',
        'compare', 'contrast', 'analyze', 'evaluate', 'discuss',
        'is', 'are', 'do', 'does', 'did', 'can', 'could', 'will', 'would',
        'should', 'may', 'might'
    }
    return question_words

def has_question_mark(text):
    """
    Checks if the text contains a question mark.
    If yes, it's likely a question.
    """
    return '?' in text

def starts_with_wh_word(text):
    """
    Checks if the text starts with a WH-word from the question vocabulary.
    This helps identify questions while avoiding false positives like 'which is interesting'.
    """
    words = text.strip().lower().split()
    if words:
        first_word = words[0]
        return first_word in get_question_vocab()
    return False

def starts_with_auxiliary(text):
    """
    Checks if the text starts with an auxiliary verb, which can indicate a question.
    """
    auxiliaries = {'is', 'are', 'do', 'does', 'did', 'can', 'could', 'will', 'would', 'should', 'may', 'might', 'am', 'was', 'were'}
    words = text.strip().lower().split()
    if words:
        first_word = words[0]
        return first_word in auxiliaries
    return False

def is_question(text):
    """
    Smart function to identify if a text is a question.
    Combines multiple checks to reduce false positives.
    """
    if has_question_mark(text):
        return True
    if starts_with_wh_word(text):
        return True
    if starts_with_auxiliary(text):
        return True
    return False

# Future feature functions can be added here
# def get_task_vocab():
#     ...

# def get_time_vocab():
#     ...