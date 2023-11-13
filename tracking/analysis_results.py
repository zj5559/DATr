import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
dataset_name = 'lasot_extension_subset'
# dataset_name = 'lasot'

"""ostrack"""
trackers.extend(trackerlist(name='ostrack', parameter_name='datr_vitb_256_mae_ce_32x4_ep300', dataset_name=dataset_name,
                            run_ids=None, display_name='datr_vitb_256'))


trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_256_mae_ce_32x4_ep300', dataset_name=dataset_name,
                            run_ids=None, display_name='vitb_256'))



dataset = get_dataset(dataset_name)
# dataset = get_dataset('otb', 'nfs', 'uav', 'tc128ce')
# plot_results(trackers, dataset, 'OTB2015', merge_results=True, plot_types=('success', 'norm_prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
print_results(trackers, dataset, dataset_name, merge_results=True,seq_eval=False, plot_types=('success', 'norm_prec', 'prec'))
# print_results(trackers, dataset, 'UNO', merge_results=True, plot_types=('success', 'prec'))
