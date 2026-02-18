from data_loading import load_subject_experiment, get_sample
import json
#file not used as a module, just used to update unique sentences label from chisco dataset

subject_nums = [1, 2, 3, 5] #4 not included because it didn't have the same number of experiments 
experiments = [10, 11, 12, 13, 14, 15] #only using 6 experiments for demo

total_trials = 0
unique_sentences = {}
output_path = r"utilities\chisco_preprocessing\unique_sentences.json"

#dont forget to modify this so it removes grammar and stuff i.e , . - ? 
for subject in subject_nums:
    print("loop subject")
    for experiment in experiments:
        print("loop experiment")

        data = load_subject_experiment(subject_num=subject, experiment_num=experiment)
        trials = len(data)
        total_trials += trials

        print("entering trial loop")

        for trial_num in range(trials):
            _, text_label_chinese, text_label_english = get_sample(data, trial_num, language='en') #this will take long because translation requires internet

            if text_label_chinese not in unique_sentences:
                unique_sentences[text_label_chinese] = text_label_english
                print("unique sentence added")

print(f"Total unique sentences collected: {len(unique_sentences)} out of {total_trials} total trials.")

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(unique_sentences, f, ensure_ascii=False, indent=4)

print("Saved!")
