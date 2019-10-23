from lib.generator import generator
from lib.reconstructor import reconstructor
from lib.discriminator_lstm import discriminator_lstm
from lib.ops import *
from utils import *
import tensorflow as tf
import numpy as np
import os
import subprocess

class seq2seq2seq():
    def __init__(self,args,sess):
        self.sess = sess
        if args.train:
            self.mode = 'all'
        elif args.pretrain:
            self.mode = 'pretrain_generator'
        else:
            self.mode = 'test'
        
        #model config
        self.rec_base = 0.0
        self.rec_weight = 0.5

        self.discriminator_iterations = 3
        self.reconstructor_iterations = 3
        self.pretrain_discriminator_steps = 501
        self.coverage_weight = 0.1
        self.lmbda = 10
        self.source_sequence_length = args.source_length
        self.code_sequence_length = args.code_length
        self.batch_size = args.batch_size
        self.num_steps =args.num_steps
        self.load_model = args.load
        self.saving_step = args.saving_step
        self.result_path = args.test_output
        self.input_path = args.test_input

        #trivial things
        self.generator_lstm_length = [self.source_sequence_length+1 for _ in range(self.batch_size)]
        self.code_lstm_length = [self.code_sequence_length+1 for _ in range(self.batch_size)] 
        self.utils = utils(args)

        #the model of generator, reconstructor, discriminator will be save in seperately directory
        self.model_dir = args.model_dir
        """
        if not os.path.exists(os.path.join(self.model_dir,'code')):
            os.makedirs(os.path.join(self.model_dir,'code'))
        if self.mode == 'all':
            subprocess.call('cp ./*.py '+ os.path.join(self.model_dir,'code'), shell=True)
        """
        self.vocab_size = self.utils.vocab_size
        self.word_embedding_dim = 300
        self.BOS = 1
        self.EOS = 0

        self.build_model()
        self.tensorflow_init()


    def tensorflow_init(self):        
        for v in tf.trainable_variables():
            print(v.name,v.get_shape().as_list())
        
        self.generator_saver = tf.train.Saver(self.generator_variables,max_to_keep=10)
        if self.mode == 'test':
            return
        self.discriminator_saver = tf.train.Saver(self.discriminator_variables,max_to_keep=2)
        self.reconstructor_saver = tf.train.Saver(self.reconstructor_variables,max_to_keep=2)


    def build_model(self):
        with tf.variable_scope("input") as scope:
            self.source_sentence = tf.placeholder(dtype=tf.int32, shape=(self.batch_size, self.source_sequence_length))
            self.reconstructor_decoder_inputs = tf.placeholder(dtype=tf.int32, shape=(self.batch_size, self.source_sequence_length))
            self.real_sample = tf.placeholder(dtype=tf.int32, shape=(self.batch_size,self.code_sequence_length))
        
            #add begin / end of sequence tag
            BOS_slice = tf.ones([self.batch_size, 1], dtype=tf.int32)*self.BOS
            EOS_slice = tf.ones([self.batch_size, 1], dtype=tf.int32)*self.EOS

            generator_decoder_inputs = tf.ones([self.batch_size,self.code_sequence_length+1],dtype=tf.int32)
            #only need when pretrain generator
            self.generator_target = tf.placeholder(dtype=tf.int32, shape=(self.batch_size,self.code_sequence_length))
            generator_target = tf.concat([self.generator_target,EOS_slice],axis=1)
            if self.mode=='pretrain_generator':
                generator_decoder_inputs = tf.concat([BOS_slice,self.generator_target],axis=1)

            reconstructor_decoder_inputs = tf.concat([BOS_slice,self.reconstructor_decoder_inputs],axis=1)
            reconstructor_decoder_targets = tf.concat([self.source_sentence,EOS_slice],axis=1)

            real_sample = tf.concat([self.real_sample,EOS_slice],axis=1)
            #real_sample = tf.one_hot(real_sample,self.vocab_size)

            global_step = tf.Variable(1, name='global_step', trainable=False, dtype=tf.float32)


        with tf.variable_scope("word_embedding") as scope:
            #word embedding for generator and reconstructor
            init = tf.contrib.layers.xavier_initializer()
            word_embedding_matrix = tf.get_variable(
                name="word_embedding_matrix",
                shape=[self.vocab_size, self.word_embedding_dim],
                initializer=init,   
                trainable = True
            )
            generator_decoder_inputs_embedded = tf.nn.embedding_lookup(word_embedding_matrix, generator_decoder_inputs)
                                        

        with tf.variable_scope("generator") as scope:
            generator_raw_output,generator_outputs_ids,generator_outputs_probs,coverage_loss = generator(
                encoder_inputs = self.source_sentence,
                vocab_size = self.vocab_size,
                word_embedding_matrix = word_embedding_matrix,
                encoder_length = self.generator_lstm_length,
                decoder_inputs = generator_decoder_inputs_embedded,
                feed_previous = False if self.mode=='pretrain_generator' else True,
                do_sample = True if self.mode=='all' else False,
                do_beam_search= True if self.mode=='test' else False
            )

            #convert to batch major
            generator_outputs = tf.stack(generator_raw_output,axis=1)
            generator_outputs_ids = tf.stop_gradient(tf.stack(generator_outputs_ids,axis=1))
            #sample_seq_len = get_seq_len(generator_outputs_ids)
            self.generator_pred = generator_outputs_ids
            self.log_p = tf.stack(generator_outputs_probs,axis=1)
            
            generator_probs = tf.reduce_max(generator_raw_output,axis=-1)
            self.generator_prob = tf.reduce_mean(generator_probs)
            if self.mode == 'test':
                self.generator_variables = [v for v in tf.trainable_variables() if v.name.startswith("generator") or v.name.startswith("word_embedding")]
                return
            
            scope.reuse_variables()

            #compute baseline loss for reconstructor
            generator_argmax_outputs,baseline_ids,_,_ = generator(
                encoder_inputs = self.source_sentence,
                vocab_size = self.vocab_size,
                word_embedding_matrix = word_embedding_matrix,
                encoder_length = self.generator_lstm_length,
                decoder_inputs = generator_decoder_inputs_embedded,
                feed_previous = True,
                do_sample = False
            )

            generator_argmax_outputs = tf.stack(generator_argmax_outputs,axis=1)
            baseline_ids = tf.stop_gradient(tf.stack(baseline_ids,axis=1))
            argmax_seq_len = get_seq_len(baseline_ids)

    
        with tf.variable_scope("reconstructor") as scope:
            reconstructor_sample_loss,reconstructor_outputs,_ = reconstructor(
                encoder_inputs =  generator_outputs_ids,
                vocab_size = self.vocab_size,
                encoder_length = self.code_lstm_length,
                decoder_inputs = reconstructor_decoder_inputs,
                decoder_targets = reconstructor_decoder_targets
            )

            scope.reuse_variables()

            reconstructor_argmax_loss,_,_= reconstructor(
                encoder_inputs = baseline_ids,
                vocab_size = self.vocab_size,
                encoder_length = self.code_lstm_length,
                decoder_inputs = reconstructor_decoder_inputs,
                decoder_targets = reconstructor_decoder_targets
            )

        with tf.variable_scope("discriminator") as scope: 
            true_sample_pred = tf.nn.sigmoid(discriminator_lstm(real_sample,self.code_lstm_length,self.vocab_size))
            scope.reuse_variables()
            false_sample_pred = tf.nn.sigmoid(discriminator_lstm(generator_outputs_ids,self.code_lstm_length,self.vocab_size))

        #Set all the variable
        self.discriminator_variables = [v for v in tf.trainable_variables() if v.name.startswith("discriminator")]
        self.generator_variables = [v for v in tf.trainable_variables() if v.name.startswith("generator") or v.name.startswith("word_embedding")]
        self.reconstructor_variables = [v for v in tf.trainable_variables() if v.name.startswith("reconstructor")]

        with tf.variable_scope("discriminator_loss") as scope:
            self.discriminator_loss = -tf.reduce_mean(tf.log(true_sample_pred + 1e-9)) - \
                tf.reduce_mean(tf.log(1. - false_sample_pred + 1e-9))

        with tf.variable_scope("reconstruct_loss") as scope:
            self.reconstruct_loss = tf.reduce_mean(reconstructor_argmax_loss)

        with tf.variable_scope("generator_loss") as scope:
            scores = []
            length = tf.cast(len(generator_outputs_probs),dtype=tf.float32)

            #reconstructor score
            rec_base = tf.maximum(self.rec_base - global_step*0.00002, 0.)
            rec_base = 0.0
            rec_weight = tf.minimum(self.rec_weight + global_step*0.00005, 1.2)
            rs = -(tf.stop_gradient(reconstructor_sample_loss) \
                - tf.stop_gradient(reconstructor_argmax_loss)) + rec_base
            reconstruct_score = rec_weight*rs
                    
            #GAN score
            for i,cur_total_score in enumerate(batch_to_time_major(false_sample_pred)):
                if i==0:
                    score = tf.stop_gradient(cur_total_score)
                else:
                    score = tf.stop_gradient(cur_total_score) #- last_score
                score = 2.*score - 1
                score = score - tf.reduce_mean(score)
                scores.append(score)
                last_score = tf.stop_gradient(cur_total_score)

            discount_scores = [[]]*len(scores)
            running_add = 0.0
            discount_rate = 0.3

            #compute total score
            for i in reversed(range(len(scores))):
                running_add = running_add*discount_rate + scores[i]
                discount_scores[i] = running_add + reconstruct_score

            total_loss = []
            total_coverage_loss = []
            for cur_score,prob,c_l in zip(discount_scores,generator_outputs_probs,coverage_loss):
                loss = tf.reduce_mean(-cur_score*tf.log(tf.clip_by_value(prob,1e-7,1.0)))
                one_coverage_loss = self.coverage_weight*tf.reduce_mean(tf.reduce_sum(c_l,axis=1))
                loss += one_coverage_loss
                total_coverage_loss.append(one_coverage_loss)
                total_loss.append(loss)

            self.generator_loss = tf.add_n(total_loss)

        with tf.variable_scope("pretrain_generator_loss") as scope:
            generator_target = batch_to_time_major(generator_target)
            total_loss = []
            total_coverage_loss = []
            length = tf.cast(len(generator_target),dtype=tf.float32)
            for prob_t,target,c_l in zip(generator_raw_output,generator_target,coverage_loss):
                target_prob = tf.reduce_max(tf.one_hot(target,self.vocab_size)*prob_t,axis=-1)
                one_coverage_loss = self.coverage_weight*tf.reduce_mean(tf.reduce_sum(c_l,axis=1))
                loss = -tf.reduce_mean(tf.log(tf.clip_by_value(target_prob,1e-9,1.0))) + 0.1*one_coverage_loss
                total_coverage_loss.append(one_coverage_loss)
                total_loss.append(loss)
            self.pretrain_generator_loss = tf.add_n(total_loss) / length
            self.pretrain_coverage_loss = tf.add_n(total_coverage_loss) / length

        with tf.variable_scope("optimizer") as scope:
            self.step_increment_op = tf.assign(global_step, global_step+1)
            if self.mode=='all':
                self.train_discriminator_op = tf.train.AdamOptimizer(0.002, beta1=0.5, beta2=0.999).minimize(
                    self.discriminator_loss, 
                    var_list=self.discriminator_variables
                )

                train_generator_op = tf.train.AdamOptimizer(0.00005, beta1=0.5, beta2=0.999)
                gradients, variables = zip(*train_generator_op.compute_gradients(self.generator_loss,\
                    var_list=self.generator_variables))
                gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                self.train_generator_op = train_generator_op.apply_gradients(zip(gradients, variables))
     
                self.train_reconstructor_op = tf.train.AdamOptimizer(0.0001).minimize(
                    self.reconstruct_loss,
                    var_list=self.reconstructor_variables
                )

            if self.mode =='pretrain_generator':
                #pretrain_generator_op = tf.train.RMSPropOptimizer(0.001)
                pretrain_generator_op = tf.train.AdamOptimizer(0.0001)

                gradients, variables = zip(*pretrain_generator_op.compute_gradients(self.pretrain_generator_loss,\
                    var_list=self.generator_variables))
                gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                self.pretrain_generator_op = pretrain_generator_op.apply_gradients(zip(gradients, variables))

        #for debug
        self.fs = false_sample_pred
        self.rs = reconstruct_score
        self.bp = baseline_ids

    
    def pretrain(self):
        step = 0
        saving_step = self.saving_step
        summary_step = int(saving_step/20)
        
        print('Start pretrain generator!!!!')

        model_dir = os.path.join(self.model_dir,'generator/')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir,'model')
        log_dir = os.path.join(model_dir,'log/')
        saver = self.generator_saver
        cur_loss = 0.0;coverage_loss=0.0;cur_prob=0.0

        self.sess.run(tf.global_variables_initializer())
        if self.load_model:
            saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))


        for x_batch,y_batch in self.utils.pretrain_generator_data_generator():
            step += 1
            feed_dict = {
                self.source_sentence:x_batch,
                self.generator_target:y_batch
            }
            #print(self.utils.id2sent(x_batch[0]))
            _,loss,c_loss,pred,prob = self.sess.run([self.pretrain_generator_op,self.pretrain_generator_loss,\
                self.pretrain_coverage_loss,self.generator_pred,self.generator_prob],feed_dict=feed_dict)
            cur_loss += loss
            coverage_loss += c_loss
            cur_prob += prob

            if step%(summary_step)==0:
                print('{step}: generator_loss: {loss} coverage_loss: {c_loss} prob: {prob}'.format(\
                     step=step,loss=cur_loss/summary_step,c_loss=coverage_loss/summary_step,prob=cur_prob/summary_step))
                #print(self.utils.id2sent(pred[0]))
                cur_loss = 0.0;coverage_loss = 0.0;cur_prob=0.0

            if step%saving_step==0:
                saver.save(self.sess, model_path, global_step=step)

            if step>=self.num_steps:
                break

        
    def train(self):
        step = 0
        saving_step = self.saving_step
        summary_step = int(saving_step/20)

        print('Start training whole model!!')

        #init model config
        self.sess.run(tf.global_variables_initializer())
        #discriminator model
        dis_model_dir = os.path.join(self.model_dir,'discriminator/')
        dis_model_path = os.path.join(dis_model_dir,'whole_model')
        if not os.path.exists(dis_model_dir):
            os.makedirs(dis_model_dir)
        #generator model
        gen_model_dir = os.path.join(self.model_dir,'generator/')
        gen_model_path = os.path.join(gen_model_dir,'whole_model')
        if not os.path.exists(gen_model_dir):
            os.makedirs(gen_model_dir)
            
        #reconstructor model
        rec_model_dir = os.path.join(self.model_dir,'reconstructor/')
        rec_model_path = os.path.join(rec_model_dir,'whole_model')
        if not os.path.exists(rec_model_dir):
            os.makedirs(rec_model_dir)

        #init loss
        gen_prob = 0.0;gen_loss = 0.0;dis_loss = 0.0;rec_loss =0.0

        self.generator_saver.restore(self.sess,tf.train.latest_checkpoint(gen_model_dir))
        if self.load_model:
            print('load model from:',self.model_dir)
            if len([f for f in os.listdir(rec_model_dir)])>0:
                print('load reconstructor')
                self.reconstructor_saver.restore(self.sess,tf.train.latest_checkpoint(rec_model_dir))
            if len([f for f in os.listdir(dis_model_dir)])>0:
                print('load discriminator')
                self.discriminator_saver.restore(self.sess,tf.train.latest_checkpoint(dis_model_dir))

        data_generator = self.utils.gan_data_generator()
        for _ in range(self.num_steps):
            step = int(self.sess.run(self.step_increment_op))

            #train discriminator
            for i in range(self.discriminator_iterations):
                source_b,real_b = data_generator.__next__()
                feed_dict = {
                    self.source_sentence:source_b,
                    self.real_sample:real_b
                }

                _,loss = self.sess.run([self.train_discriminator_op,self.discriminator_loss],feed_dict=feed_dict)
                dis_loss += loss/self.discriminator_iterations
            
            #train reconstructor only
            if step<self.pretrain_discriminator_steps:
                for i in range(self.reconstructor_iterations):
                    source_b,real_b = data_generator.__next__()
                    feed_dict = {
                        self.source_sentence:source_b,
                        self.reconstructor_decoder_inputs:source_b
                    }

                    _,loss = self.sess.run([self.train_reconstructor_op,self.reconstruct_loss],feed_dict=feed_dict)
                    rec_loss += loss / self.reconstructor_iterations
            

            if step>=self.pretrain_discriminator_steps:
                #train generator only
                source_b,real_b = data_generator.__next__()
                feed_dict = {
                    self.source_sentence:source_b,
                    self.reconstructor_decoder_inputs:source_b,
                }

                _,_,r_l,loss,prob,pred,fs,rs,bp = self.sess.run([self.train_generator_op,
                    self.train_reconstructor_op,self.reconstruct_loss,self.generator_loss,
                    self.generator_prob,self.generator_pred,self.fs,self.rs,self.bp],feed_dict=feed_dict)

                rec_loss += r_l
                gen_prob += prob
                gen_loss += loss

            #make summary
            if step%(summary_step)==0:
                print('{step}: dis_loss: {dis_loss} gen_loss: {gen_loss} gen_prob: {gen_prob} rec_loss: {rec_loss}'.format(
                    step=step,dis_loss=dis_loss/summary_step,gen_loss=gen_loss/summary_step,gen_prob=gen_prob/summary_step,rec_loss=rec_loss/summary_step))
                """
                if step>=self.pretrain_discriminator_steps:
                    print('sample:',self.utils.id2sent(pred[0]))
                    print('argmax:',self.utils.id2sent(bp[0]))
                    print('false score:',fs[0])
                    print('rec_score:',rs[0])
                """
                gen_prob = 0.0;gen_loss = 0.0;dis_loss = 0.0;rec_loss =0.0
                
            if step%saving_step==0:
                print('saving model!!!!......')
                self.discriminator_saver.save(self.sess, dis_model_path, global_step=step)
                self.generator_saver.save(self.sess, gen_model_path, global_step=step)
                self.reconstructor_saver.save(self.sess, rec_model_path, global_step=step)

            if step>=self.num_steps:
                break


    def test(self):
        result = open('result.txt','w')
        self.sess.run(tf.global_variables_initializer())
        gen_model_dir = os.path.join(self.model_dir,'generator/')
        self.generator_saver.restore(self.sess,tf.train.latest_checkpoint(gen_model_dir))
        print('loading model from',self.model_dir)
        count = 0
        pp = []
        max_len = len(open(self.input_path).readlines())
        for x_batch in self.utils.test_data_generator(self.input_path):
            feed_dict = {
                self.source_sentence:x_batch
            }
            raw_pred,prob = self.sess.run([self.generator_pred,self.generator_prob],feed_dict=feed_dict)
            pp.append(prob)
            for i in range(len(raw_pred)):
                last_id = 0
                pred = [[]]*len(raw_pred[i])
                for j in reversed(range(len(raw_pred[i]))):
                    pred[j] = raw_pred[i][j][last_id] % self.vocab_size
                    last_id = int(raw_pred[i][j][last_id] / self.vocab_size)
                result.write(self.utils.id2sent(pred) + '\n')
                count += 1
                if count>=1933:
                    break
            if count>=1933:
                break
        print(np.mean(pp))
        print('finishing testing!!!!!')