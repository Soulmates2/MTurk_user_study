import argparse


parser = argparse.ArgumentParser(description='Generate the HTML for a HIT')

# In a single HIT, how many comparisons will the Turker make?
parser.add_argument('--n_real_comparisons_per_hit', type=int, default=20)
# How many "vigilance tests" (i.e. 'obvious' comparisons to check if the Turker is paying attention) per HIT?
parser.add_argument('--n_vigilance_tests_per_hit', type=int, default=5)
# The base URL where the experiment images live
# For the scene synthesis work, we used an Amazon S3 bucket: https://s3.us-east-2.amazonaws.com/fast-synth-study/
parser.add_argument('--base_url', type=str)
# The description of the experiment that is shown to Turkers
# For example, for the scene synthesis work, we used the following description:
# "You will take part in an experiment involving visual perception (~ 5 min).
#  You will be shown pairs of images depicting virtual 3D rooms and asked to choose the one you think is more plausible"
parser.add_argument('--experiment_description', type=str)
# The prompt text that is shown above every force-choice comparison.
# For example, for the scene synthesis work, we used the following prompt:
# "Here are two images depicting the same type of virtual 3D room. Which of these two rooms you do think is more plausible?"
parser.add_argument('--prompt', type=str)
# Name of the output HTML file to produce
parser.add_argument('--out_filename', type=str, default='hit.html')

args = parser.parse_args()



args.n_real_comparisons_per_hit = 20
args.n_vigilance_tests_per_hit = 5
args.base_url = 'https://as-plausible-as-possible.s3.us-east-2.amazonaws.com/'
args.experiment_description = 'You will take part in an experiment involving visual perception (~ 5 min). You will be shown sets of images and asked to answer which image looks more natural.'
args.prompt = 'Red circles indicate the source handles and Green circles indicate the target handles. We want to edit images moving the source handles to target handles. Which one appears the most plausible edited image to you? Choose one of the following images.'
args.out_filename = 'APAP_choose_multi_plausible.html'



n_comparisons = args.n_real_comparisons_per_hit + args.n_vigilance_tests_per_hit


with open('hit_template.html', 'r') as f:
    hit_template = f.read()

hit_template = hit_template.replace('__NUM_REAL_COMPARISONS__', str(args.n_real_comparisons_per_hit))
hit_template = hit_template.replace('__NUM_VIGILANCE_TESTS__', str(args.n_vigilance_tests_per_hit))
hit_template = hit_template.replace('__IMG_BASE_URL__', f"'{args.base_url}'")
hit_template = hit_template.replace('__EXPERIMENT_DESCRIPTION__', args.experiment_description)
hit_template = hit_template.replace('__PROMPT__', f"'{args.prompt}'")

sequence_helpers = ''
for i in range(1, n_comparisons+1):
    sequence_helpers += 'sequence_helper("${gt_side%d}","${images_left%d}","${images_right%d}");\n' % (i, i, i)
hit_template = hit_template.replace('__SEQUENCE_HELPERS__', sequence_helpers)

hidden_inputs = ''
for i in range(1, n_comparisons+1):
    hidden_inputs += '<input id="selection%d" name="selection%d" type="hidden" value="unset" />\n' % (i, i)
hit_template = hit_template.replace('__HIDDEN_INPUTS__', hidden_inputs)

with open(args.out_filename, 'w') as f:
    f.write(hit_template)

