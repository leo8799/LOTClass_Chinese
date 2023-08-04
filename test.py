from src.trainer import LOTClassTrainer
from config.configs_interface import configs as args
trainer = LOTClassTrainer(args)
trainer.category_vocabulary(top_pred_num=args.train_args.top_pred_num,
                            category_vocab_size=args.train_args.category_vocab_size)
trainer.self_train(epochs=args.train_args.SELF_TRAIN_EPOCH, loader_name=args.data.final_model)
