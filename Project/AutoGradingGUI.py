import time
import json
import tkinter as tk
import tkinter.font as font
from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import askopenfile
from grading import *


master = tk.Tk()
master.geometry("100000x100000")
master.configure(bg='light blue')
master.title("Automated Answer Grading")

tk.Label(master, text="Dataset", width=10, height=2, bg='grey', fg='dark blue', font=20).grid(row=0)
tk.Label(master, bg="black").grid(row=11, column=1)

training_label = tk.Label(master, text="Training", width=10, height=2, bg='grey', fg='dark blue', font=20)
training_label.grid(row=5)

progress_bar_train = Progressbar(master, orient="horizontal", length=1000, mode="determinate")
progress_bar_train.grid(column=1, row=5, pady=15)

progress_bar_grade = Progressbar(master, orient="horizontal", length=1000, mode="determinate")
progress_bar_grade.grid(column=1, row=20, pady=15)

grading_label = tk.Label(master, text="Grading", width=10, height=2, bg='grey', fg='dark blue', font=20)
grading_label.grid(row=20)

tk.Label(master, text="QA", width=10, height=2, bg='grey', fg='dark blue', font=20).grid(row=7)
tk.Label(master, text="OUTPUT", width=10, height=2, bg='grey', fg='dark blue', font=20).grid(row=50)

entry_text_dataset = tk.StringVar()
entry_text_QA = tk.StringVar()
entry_text_output = tk.StringVar()

entry_dataset = tk.Entry(master, textvariable=entry_text_dataset, width=50, font=20)
entry_QA = tk.Entry(master, textvariable=entry_text_QA, width=50, font=20)
entry_output = tk.Text(master, width=50, height=10)

entry_dataset.grid(row=0, column=1)
entry_QA.grid(row=7, column=1)
entry_output.grid(row=50, column=1)

questions = []
models = {}
processed_dataset = {}
evaluation = {}

def file_select(entry_text):
    file = askopenfile(mode='r', filetypes=[('All Files', '*.*')])
    entry_text.set(file.name)

def run_train():
    train_dataset = entry_text_dataset.get()
    global models
    global processed_dataset
    
    # Set the maximum value of progress bar to 200
    progress_bar_train['maximum'] = 200
    
    # Process the dataset
    with open(train_dataset) as f:
        dataset = json.load(f)
    processed_dataset = prepare_dataset(dataset)
    questions = list(processed_dataset)
    
    # Update progress bar value to 100 for dataset processing
    progress_bar_train["value"] = 100
    progress_bar_train.update()
    
    # Build models
    models = build_models(processed_dataset)
    
    # Update progress bar value to 200 for model building
    progress_bar_train["value"] = 200
    progress_bar_train.update()
        
def run_grade():
    test_file = entry_text_QA.get()
    global evaluation
    
    # Set the maximum value of progress bar to 200
    progress_bar_grade['maximum'] = 200
    
    # Read the test file and prepare student's QA
    QA_dataset = read_dataset(test_file)
    student_QA = prepare_QA(QA_dataset)
    
    # Update progress bar value to 100
    progress_bar_grade["value"] = 100
    progress_bar_grade.update()

    # Evaluate the student's QA against the models
    evaluation = evaluate_QA(questions, student_QA, models, processed_dataset)

    progress_bar_grade['value'] = 200
    progress_bar_grade.update()
    
    
def show_result():
    marks_secured = 0
    marks_allotted = 0
    for question, data in evaluation.items():
        marks_secured += data['marks_awarded']
        marks_allotted += data['max_marks']
        print("Question:", question)
        print("Answer:", data['answer'])
        print("Marks Awarded:", data['marks_awarded'])
        print("Max Marks:", data['max_marks'])
        print()
    marks = (marks_secured, marks_allotted)
    entry_output.insert(tk.END, f"Total Marks Secured: {round(marks_secured, 2)} / {marks_allotted}\n")
    entry_output.insert(tk.END, f"Percentage Marks Secured: {round(100*marks_secured/marks_allotted, 2)}%.\n")

select_button_dataset = tk.Button(master, text='Select', width=10, height=2, bg='black', fg='white', font=50,
command=lambda: file_select(entry_text_dataset)).grid(row=0, column=3, pady=4)

select_button_QA = tk.Button(master, text='Select', width=10, height=2, bg='black', fg='white', font=50,
command=lambda: file_select(entry_text_QA)).grid(row=7, column=3, pady=4)

display_marks_button = tk.Button(master, text='Display Marks', width=15, height=2, bg='black', fg='white', font=20,
command=show_result).grid(row=50, column=3, pady=4)

grade_button = tk.Button(master, text='Grade', width=10, height=2, bg='black', fg='white', font=50,
command=run_grade).grid(row=9, column=1, pady=4)

train_button = tk.Button(master, text='Train', width=10, height=2, bg='black', fg='white', font=50,
command=run_train).grid(row=3, column=1, pady=4)

tk.mainloop()