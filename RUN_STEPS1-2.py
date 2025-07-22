import runpy

global_cohort_vars= {'cohort_name' : '250630_cohort',
                      'cohort_mice_csv' : '250630_cohort_mouse_info.csv'}

runpy.run_path('Step1_cohort_info.py',
               init_globals=global_cohort_vars)
runpy.run_path('Step2_cohort_align.py',
               init_globals=global_cohort_vars)

global_cohort_vars = {'cohort_name' : '250620_HL_cohort',
                      'cohort_mice_csv' : '250620_HL_cohort_mouse_info.csv'}

runpy.run_path('Step1_cohort_info.py',
               init_globals=global_cohort_vars)
runpy.run_path('Step2_cohort_align.py',
               init_globals=global_cohort_vars)