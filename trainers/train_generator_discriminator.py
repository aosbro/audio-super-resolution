class GeneratorDiscriminatorTrainer:
    def __init__(self, model, train_generator, test_generator, valid_generator, lr, savepath):
        # Models
        self.model = model

        # Data generators
        self.train_generator = train_generator
        self.test_generator = test_generator
        self.valid_generator = valid_generator

        # Optimizer
        self.optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

        # Device
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')

        # Loss function and stored losses
        self.loss_function = nn.MSELoss()
        self.train_losses = []
        self.test_losses = []
        self.valid_losses = []

        # Path to save to the class
        self.savepath = savepath