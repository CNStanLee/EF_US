from pruning import *

def auto_prune_model_bottle(ori_model, model_name, w, a, dataset_name='MNIST', pruning_type='l1', new_sensitivity=False):

    accuracy_drop_tolerance = 5 # 5  30 is for fast convergence and test, 5 is for real case
    final_accuracy_drop_tolerance = 3  # 1% final accuracy drop tolerance
    retrain_epochs = 10  # number of epochs to retrain the model after pruning
    final_retrain_epochs = 200  # number of epochs to retrain the final model after pruning
    sensitivity_step = 1    # step size for sensitivity analysis, 1% for each step
    tolerance_step = 1  # step size for increasing accuracy drop tolerance

    converge = False
    # load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ori_model.to(device) 
    # Stage 1: sensitivity analysis
    print("\nPruning Stage 1: Sensitivity Analysis")
    sparsity_info = analyze_model_sparsity(ori_model)
    # init
    torch.manual_seed(1998)
    torch.cuda.manual_seed(1998)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  
    model = copy.deepcopy(ori_model).to(device)
    train_loader, val_loader, test_loader = get_dataloaders(dataset_name)
    original_acc = 0
    for i in range(10):
        original_acc_1 = test(model, test_loader, device)
        original_acc += original_acc_1
    original_acc /= 10
    print(f"Original Test Accuracy: {original_acc:.2f}%")
    
    sensitivity_result_name = model_name + f'_w{w}_a{a}_sensitivity_results.csv'
    if not os.path.exists(sensitivity_result_name):
        sensitivity_results = sensitivity_analysis(model, test_loader, sparsity_info, pruning_type=pruning_type, step = sensitivity_step)
        save_to_csv(sensitivity_results, sensitivity_result_name)
    elif new_sensitivity:
        # delete the old sensitivity results
        print(f"Old sensitivity results {sensitivity_result_name} exists, deleting it and rerunning sensitivity analysis.")
        os.remove(sensitivity_result_name)
        print(f"Rerunning sensitivity analysis for model {model_name} with pruning type {pruning_type}.")
        sensitivity_results = sensitivity_analysis(model, test_loader, sparsity_info, pruning_type=pruning_type, step = sensitivity_step)
        save_to_csv(sensitivity_results, sensitivity_result_name)
    else:
        sensitivity_results = load_from_csv(sensitivity_result_name)
    # Stage 2: sparsity first decision
        converge = False
    print("\nPruning Stage 2: Pruning with Specific Rates")
    # final_model = copy.deepcopy(model)
    got_final_model = False
    while not converge:
        pruning_decisions = determine_safe_pruning_rates(sensitivity_results, accuracy_drop_tolerance)
        # Print results
        for decision in pruning_decisions:
            print(f"Layer {decision.layer_path} - Safe pruning rate: {decision.pruning_rate*100:.1f}%")

        pruned_model = prune_model_by_decisions(model, pruning_decisions, pruning_type=pruning_type)
        # Stage 3: Re-train the model, try to fit accuracy dropdown acceptable
        print("\nPruning Stage 3: Re-training the Pruned Model")
        retrained_model = retrain_model(pruned_model, train_loader, val_loader, epochs=retrain_epochs)
        analyze_model_sparsity(retrained_model)
        # Stage 4: if accuracy is not acceptable, try to change tolerance and pruning rate, do 4 again
        # find 10 times average accuracy
        test_acc = 0
        for i in range(10):
            test_acc_1 = test(retrained_model, test_loader, device)
            test_acc += test_acc_1
        test_acc /= 10

        #final_model = copy.deepcopy(retrained_model)
        print(f"Test Accuracy after re-training: {test_acc:.2f}%")
        accuracy_drop = original_acc - test_acc
        print(f"Accuracy drop after pruning and re-training: {accuracy_drop:.2f}%")
        if accuracy_drop <= final_accuracy_drop_tolerance:
            print("Final model is acceptable, try bigger accuracy_drop_tolerance")
            accuracy_drop_tolerance += tolerance_step
            print(f"Current layer accuracy dropdown tolerance: {accuracy_drop_tolerance}" )
            final_model = copy.deepcopy(retrained_model)
            got_final_model = True
        else:
            # Increase the tolerance for the next iteration
            converge = True
            if not got_final_model:
                print('first try not converge')
                final_model = copy.deepcopy(retrained_model)
            print(f"end.")
        # final_model = copy.deepcopy(retrained_model)
    test_acc = test(final_model, test_loader, device)
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print("\nPruning Stage 5: Final Retraining")
    final_model2 = retrain_model(final_model, train_loader, val_loader, epochs=final_retrain_epochs)
    test_acc = test(final_model2, test_loader, device)
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    analyze_model_sparsity(final_model2)
    # Stage 3: generate estimation
    # Stage 4: Latency, resource estimation
    # Stage 5: Sparse and unfold decision
    # Stage 6: Excute pruning


def auto_prune_model_new(ori_model, model_name, w, a, dataset_name='MNIST', pruning_type='l1'):
    print(f"Auto pruning model {model_name} with weight bit width {w} and activation bit width {a} using {pruning_type} pruning.")
    if pruning_type == 'ram':
        print('ram pruning')
        model = auto_ram_pruning(model = ori_model, show_graph = False, model_name = model_name, w = w, a = a, pruning_type = pruning_type)
    else:
        print('pruning with sens')
        model = auto_prune_model_bottle(ori_model = ori_model,
                                         model_name = model_name,
                                           w = w,
                                             a = a,
                                               dataset_name=dataset_name,
                                                 pruning_type = pruning_type,
                                                 new_sensitivity=False)
    return model


def main():
    print("Bottleneck pruning script is running.")
    w = 1
    a = 1
    model_name = "2c3f_relu"
    model_weight = "./model/best_2c3f_w1_a1_500.pth"
    ori_model = get_model(model_name, w, a)
    ori_model.load_state_dict(torch.load(model_weight))
    ori_model.to(device)
    auto_prune_model_new(ori_model, model_name, w, a, dataset_name='MNIST', pruning_type='l1')

if __name__ == "__main__":
    main()

