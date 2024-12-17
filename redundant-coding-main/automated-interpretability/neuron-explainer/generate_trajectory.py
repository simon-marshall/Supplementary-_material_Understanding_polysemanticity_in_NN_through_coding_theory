# %%
import asyncio
import tiktoken

import numpy as np
import pylab as plt

from neuron_explainer.activations.activation_records import calculate_max_activation
from neuron_explainer.activations.activations import ActivationRecord
from neuron_explainer.explanations.calibrated_simulator import UncalibratedNeuronSimulator
from neuron_explainer.explanations.explainer import TokenActivationPairExplainer
from neuron_explainer.explanations.prompt_builder import PromptFormat
from neuron_explainer.explanations.scoring import simulate_and_score
from neuron_explainer.explanations.simulator import ExplanationNeuronSimulator

from datetime import datetime

# set parameters
EXPLAINER_MODEL_NAME : str = "gpt-4"
SIMULATOR_MODEL_NAME : str = "text-davinci-003"

tokenizer = tiktoken.get_encoding("gpt2")

# Load a activation projection
for ii in range(1000):
    # Load a neuron record
    random_projection = np.load(f'./data/gpt2-xl/random_projection_xl_{ii}.npy', allow_pickle=True)

    # Merge the activations into a single dictionary
    random_projection_ = {}
    prompts = []
    for rp in random_projection:
        for k,v in rp.items():
            if k == 'text':
                prompts.append(v)
                continue
            v = v.squeeze()
            if len(v) < 64:
                v = np.concatenate([v, np.zeros(64-len(v))])
            if k not in random_projection_:
                random_projection_[k] = v[:64][np.newaxis,:]
            else:
                random_projection_[k] = np.concatenate([random_projection_[k], v[:64][np.newaxis,:]], axis=0)
    # convert to activation records
    all_act_records = {}
    for k , v in random_projection_.items():
        all_act_records[k] = []
        for vv , prompt in zip(v, prompts):
            etoks = [tokenizer.decode([ek]) for ek in tokenizer.encode(prompt)[:64]]
            if len(etoks) < 64:
                etoks += [''] * (64 - len(etoks))
            assert len(etoks) == len(vv), f"{len(etoks)} != {len(vv)}"
            all_act_records[k].append(ActivationRecord(tokens=etoks, activations=vv))

    def get_top_activations(activation_records):
        """
        The function returns the top 10 activation records based on their mean maximum activation value
        compared to all activation records.
        
        :param activation_records: It is a list of activation records. Each activation record contains
        information about the activations of a particular layer in a neural network for a particular
        input. The activations are represented as a numpy array
        :return: The function `get_top_activations` returns a list of the top 10 activation records
        based on the mean of the maximum activations of each record compared to all the activations in
        the input `activation_records`.
        """
        dist_all = [ar.activations for ar in np.array(activation_records).ravel()]
        return sorted(activation_records, key=lambda x: np.mean(np.max(x.activations) > dist_all), reverse=True)[:10]

    # Generate explanations for the top 10 activation records
    explanations_and_scores = {}
    for c_layer in all_act_records.keys():
        top_activating = get_top_activations(all_act_records[c_layer])
        train_activation_records = top_activating[::2]
        valid_activation_records = top_activating[1::2]

        async def main():
            # Generate an explanation for the neuron.
            explainer = TokenActivationPairExplainer(
                model_name=EXPLAINER_MODEL_NAME,
                prompt_format=PromptFormat.HARMONY_V4,
                max_concurrent=1,
            )
            explanations = await explainer.generate_explanations(
                all_activation_records=train_activation_records,
                max_activation=calculate_max_activation(train_activation_records),
                num_samples=1,
            )
            assert len(explanations) == 1
            explanation = explanations[0]
            print(f"{explanation=}")

            # Simulate and score the explanation.
            simulator = UncalibratedNeuronSimulator(
                ExplanationNeuronSimulator(
                    SIMULATOR_MODEL_NAME,
                    explanation,
                    max_concurrent=1,
                    prompt_format=PromptFormat.INSTRUCTION_FOLLOWING,
                )
            )
            scored_simulation = await simulate_and_score(simulator, valid_activation_records)
            print(f"score={scored_simulation.get_preferred_score():.2f}")
            explanations_and_scores[c_layer] = (explanation, scored_simulation.get_preferred_score())

        # Run the main function using asyncio
        asyncio.run(main())

    # save the explanations and scores
    c_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    np.save(f'./data/gpt2-large-full/explanations_and_scores-{c_time}.npy', explanations_and_scores, allow_pickle=True)

    plt.figure(figsize=(10,10))
    plt.plot([v[1] for v in explanations_and_scores.values()])
    plt.savefig(f'./data/gpt2-large-full/explanations_and_scores_{ii}.pdf')
