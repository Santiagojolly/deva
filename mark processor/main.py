import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import rankdata

# --- Session State Setup ---
if "students" not in st.session_state:
    st.session_state.students = []

if "subject_names" not in st.session_state:
    st.session_state.subject_names = ["Subject 1", "Subject 2"]

if "sem_count" not in st.session_state:
    st.session_state.sem_count = 2

# --- Sidebar Configuration ---
st.sidebar.header("Configuration")

num_subjects = st.sidebar.number_input(
    "Number of Subjects", min_value=1, max_value=10,
    value=len(st.session_state.subject_names), step=1
)

# Rename subjects
new_subject_names = []
for i in range(num_subjects):
    name = st.sidebar.text_input(
        f"Subject {i + 1} Name",
        value=st.session_state.subject_names[i] if i < len(st.session_state.subject_names) else f"Subject {i + 1}"
    )
    new_subject_names.append(name)
st.session_state.subject_names = new_subject_names

# Semester count
sem_input = st.sidebar.number_input(
    "Total Semesters (including current)", min_value=2, max_value=10,
    value=st.session_state.sem_count
)
st.session_state.sem_count = sem_input

# --- Student Input ---
st.subheader("Enter Student Data")

num_students = st.number_input(
    "Number of Students", min_value=1, max_value=100,
    value=len(st.session_state.students) or 1, step=1
)

# Resize student list
while len(st.session_state.students) < num_students:
    st.session_state.students.append({
        "name": "",
        "roll": "",
        "marks": [[0.0] * num_subjects for _ in range(st.session_state.sem_count)]
    })

st.session_state.students = st.session_state.students[:num_students]

# Resize marks for each student
for student in st.session_state.students:
    while len(student["marks"]) < st.session_state.sem_count:
        student["marks"].append([0.0] * num_subjects)
    student["marks"] = student["marks"][:st.session_state.sem_count]
    for sem in range(st.session_state.sem_count):
        if len(student["marks"][sem]) < num_subjects:
            student["marks"][sem] += [0.0] * (num_subjects - len(student["marks"][sem]))
        elif len(student["marks"][sem]) > num_subjects:
            student["marks"][sem] = student["marks"][sem][:num_subjects]

# --- Student Form ---
for idx in range(num_students):
    student = st.session_state.students[idx]
    with st.expander(f"Student {idx + 1}"):
        student["name"] = st.text_input("Name", value=student["name"], key=f"name_{idx}")
        student["roll"] = st.text_input("Roll No.", value=student["roll"], key=f"roll_{idx}")

        for sem in range(st.session_state.sem_count):
            st.markdown(f"**Semester {sem + 1} Marks**")
            cols = st.columns(num_subjects)
            for subj in range(num_subjects):
                student["marks"][sem][subj] = cols[subj].number_input(
                    label=st.session_state.subject_names[subj],
                    min_value=0.0, max_value=100.0,
                    value=student["marks"][sem][subj],
                    key=f"s{idx}_sem{sem}_subj{subj}"
                )

# --- Utility Functions ---
def compute_results(marks):
    total = np.sum(marks, axis=1)
    average = np.mean(marks, axis=1)
    grades = []
    for avg in average:
        if avg >= 90:
            grades.append("A")
        elif avg >= 75:
            grades.append("B")
        elif avg >= 60:
            grades.append("C")
        elif avg >= 40:
            grades.append("D")
        else:
            grades.append("F")
    return total, average, grades

def assign_ranks(total):
    return rankdata(-total, method="min").astype(int)

def predict_next_semester(all_sem_marks):
    predictions = []
    for student in all_sem_marks:
        student_prediction = []
        for subj in range(num_subjects):
            X, y = [], []
            for sem_index in range(len(student)):
                try:
                    mark = student[sem_index][subj]
                    X.append([sem_index + 1])
                    y.append(mark)
                except IndexError:
                    continue

            if len(X) < 2:
                student_prediction.append(y[-1] if y else 0)
                continue

            model = LinearRegression()
            model.fit(X, y)
            pred = model.predict([[len(student) + 1]])[0]
            student_prediction.append(max(0, min(100, pred)))
        predictions.append(student_prediction)
    return predictions

# --- Report Generation ---
if st.button("📊 Generate Current Semester Report"):
    names = [s["name"] for s in st.session_state.students]
    rolls = [s["roll"] for s in st.session_state.students]
    marks = [s["marks"] for s in st.session_state.students]

    if any(not name or not roll for name, roll in zip(names, rolls)):
        st.warning("⚠️ Please fill in all student names and roll numbers.")
    else:
        current_sem_idx = st.session_state.sem_count - 1
        current_sem_marks = np.array([s["marks"][current_sem_idx] for s in st.session_state.students])

        total, avg, grades = compute_results(current_sem_marks)
        ranks = assign_ranks(total)
        predictions = predict_next_semester(marks)

        headers = ["Name", "Roll"] + st.session_state.subject_names + ["Grade", "Rank"] + [f"Pred {s}" for s in st.session_state.subject_names]
        table_data = []

        for i in range(len(names)):
            row = [names[i], rolls[i]] + list(current_sem_marks[i]) + [grades[i], ranks[i]] + [f"{p:.2f}" for p in predictions[i]]
            table_data.append(row)

        df = pd.DataFrame(table_data, columns=headers)

        st.subheader("✅ Current Semester Report")
        st.dataframe(df, use_container_width=True)

        # CSV download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Report as CSV", data=csv, file_name="semester_report.csv", mime="text/csv")

# --- Graphs ---
st.subheader("📈 Subject-wise Progress Graphs (with Prediction)")

for i, student in enumerate(st.session_state.students):
    st.markdown(f"### 👤 {student['name']} ({student['roll']})")
    fig, ax = plt.subplots(figsize=(10, 3))

    for subj_index, subj_name in enumerate(st.session_state.subject_names):
        subj_marks = [student["marks"][sem][subj_index] for sem in range(st.session_state.sem_count)]
        predicted_score = predict_next_semester([student["marks"]])[0][subj_index]

        sem_x = list(range(1, st.session_state.sem_count + 1))
        sem_x_extended = sem_x + [sem_x[-1] + 1]
        subj_marks_extended = subj_marks + [predicted_score]

        ax.plot(sem_x, subj_marks, marker='o', label=f"{subj_name} (Actual)")
        ax.plot(sem_x_extended[-2:], subj_marks_extended[-2:], marker='x', linestyle='--', label=f"{subj_name} (Predicted)")

    ax.set_xlabel("Semester")
    ax.set_ylabel("Marks")
    ax.set_ylim(0, 100)
    ax.set_title("Subject-wise Progress with Prediction")
    ax.set_xticks(sem_x_extended)
    ax.legend(loc="upper left", fontsize="small")
    st.pyplot(fig)

st.success("✅ Report and Graphs Generated Successfully!")
