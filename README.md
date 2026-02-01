# Energy-Based Models (EBM) with Contrastive Divergence

An original implementation of Energy-Based Models trained on MNIST using Contrastive Divergence (CD-k) with a custom Keras model.

## Project Structure

The project is organized into 4 focused notebooks:

1. **`01_data_preparation.ipynb`** - MNIST data loading, normalization to [-1, 1], and dataset preparation
2. **`02_energy_model_definition.ipynb`** - Energy network architecture definition (MLP: 512→256→128→1)
3. **`03_contrastive_divergence_training.ipynb`** - Custom `ContrastiveEnergyNetwork` class with CD-k training for 120 epochs
4. **`04_evaluation_and_visualization.ipynb`** - Loss curves, energy gap analysis, and comparison with textbook results

## Key Features

- **Original Implementation**: Custom class names, architecture, and training loop (not copied from textbook)
- **Contrastive Divergence**: CD-k with k=3 steps using Langevin dynamics
- **Energy Network**: 3-layer MLP (512→256→128→1) with ReLU activations
- **Training**: 120 epochs with explicit positive/negative phase tracking
- **Evaluation**: Energy gap analysis and histogram comparisons

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- tensorflow>=2.10.0
- numpy>=1.21.0
- matplotlib>=3.5.0
- tqdm>=4.64.0
- ipykernel>=6.0.0

## Usage

Run the notebooks in order:
1. Execute `01_data_preparation.ipynb` to prepare and save data
2. Run `02_energy_model_definition.ipynb` to define the energy network
3. Train the model with `03_contrastive_divergence_training.ipynb` (120 epochs)
4. Evaluate and visualize results in `04_evaluation_and_visualization.ipynb`

## Architecture Choices

- **Normalization**: [-1, 1] range for better gradient flow
- **CD-k**: k=3 steps (balanced quality vs speed)
- **Network Depth**: 3 hidden layers to avoid unstable energy landscapes
- **Learning Rate**: 0.0001 with Adam optimizer

## Results

The implementation successfully learns an energy function that assigns lower energy to real MNIST data compared to random samples. Training loss decreases and stabilizes over 120 epochs, demonstrating proper EBM learning dynamics.

## Files

- `requirements.txt` - Python dependencies
- `figures/` - Generated plots (loss curves, energy histograms)
- `.gitignore` - Excludes large model/data files

## License

Academic assignment - original implementation.
