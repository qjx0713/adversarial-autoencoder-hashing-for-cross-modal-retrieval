from setting import *
from tnet import *
import tensorflow as tf
from ops import *
from calc_hammingranking import calc_map
import os
import scipy.io as sio
from tqdm import tqdm
from numpy import *

class MODEL(object):
    def __init__(self, sess):

        self.train_L = train_L
        self.train_X = train_x
        self.train_Y = train_y

        self.query_L = query_L
        self.query_X = query_x
        self.query_Y = query_y

        self.retrieval_L = retrieval_L
        self.retrieval_X = retrieval_x
        self.retrieval_Y = retrieval_y

        self.lr_lab = lr_lab
        self.lr_img = lr_img
        self.lr_txt = lr_txt
        self.lr_dis = lr_dis
        self.Sim = Sim

        self.meanpix = imean
        self.lab_net = lab_net
        self.img_net = img_net
        self.txt_net = txt_net
        self.dis_net_IL = dis_net_IL
        self.dis_net_TL = dis_net_TL
        self.dis_net_IAE = dis_net_IAE
        self.dis_net_TAE = dis_net_TAE

        self.mse_loss = mse_criterion
        self.sce_loss = sce_criterion

        self.AE_FEA = AE_FEA
        self.image_size = image_size
        self.numClass = numClass
        self.dimText = dimText
        self.dimLab = dimLab
        self.phase = phase
        self.checkpoint_dir = checkpoint_dir
        self.dataset_dir = dataset_dir
        self.bit = bit
        self.num_train = num_train
        self.batch_size = batch_size
        self.build_model()
        self.saver = tf.train.Saver()
        self.sess = sess

    def build_model(self):
        self.ph = {}
        self.ph['label_input'] = tf.placeholder(tf.float32, (None, 1, self.numClass, 1), name='label_input')
        self.ph['image_input'] = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='image_input')
        self.ph['text_input'] = tf.placeholder(tf.float32, [None, 1, self.dimText, 1], name='text_input')
        self.ph['lr_hash'] = tf.placeholder('float32', (), name='lr_hash')
        self.ph['lr_lab'] = tf.placeholder('float32', (), name='lr_lab')
        self.ph['lr_img'] = tf.placeholder('float32', (), name='lr_img')
        self.ph['lr_txt'] = tf.placeholder('float32', (), name='lr_txt')
        self.ph['lr_dis'] = tf.placeholder('float32', (), name='lr_discriminator')
        self.ph['keep_prob'] = tf.placeholder('float32', (), name='keep_prob')
        self.ph['Sim'] = tf.placeholder('float32', [self.num_train, self.batch_size], name='Sim')
        self.ph['F'] = tf.placeholder('float32', [None, self.bit], name='F')
        self.ph['G'] = tf.placeholder('float32', [None, self.bit], name='G')
        self.ph['H'] = tf.placeholder('float32', [None, self.bit], name='H')
        self.ph['L_batch'] = tf.placeholder('float32', [None, self.numClass], name='L_batch')
        self.ph['B_batch'] = tf.placeholder('float32', [None, self.bit], name='b_batch')

        self.ph['IAE_OUT_batch'] = tf.placeholder('float32', [None,  1, self.AE_FEA, 1], name='IAE_OUT_batch')
        self.ph['TAE_OUT_batch'] = tf.placeholder('float32', [None, 1,  self.AE_FEA, 1], name='TAE_OUT_batch')
        self.ph['IAE_IN_batch'] = tf.placeholder('float32', [None, 1,  self.AE_FEA, 1], name='IAE_IN_batch')
        self.ph['TAE_IN_batch'] = tf.placeholder('float32', [None, 1, self.AE_FEA, 1], name='TAE_IN_batch')

        self.ph['L_hash_batch'] = tf.placeholder('float32', [None,1, self.bit, 1], name='L_hash_batch')
        self.ph['I_hash_batch'] = tf.placeholder('float32', [None, 1, self.bit, 1], name='I_hash_batch')
        self.ph['T_hash_batch'] = tf.placeholder('float32', [None, 1, self.bit, 1], name='T_hash_batch')

        self.ph['I_pos'] = tf.placeholder('float32', [None,  self.bit],name='I_pos')
        self.ph['I_neg'] = tf.placeholder('float32', [None,  self.bit], name='I_neg')
        self.ph['T_pos'] = tf.placeholder('float32', [None,  self.bit], name='T_pos')
        self.ph['T_neg'] = tf.placeholder('float32', [None,  self.bit], name='T_neg')

        self.ph['label_L'] = tf.placeholder('float32', [None, self.numClass], name='label_L')

            # construct label network
        self.Hsh_L, self.Label_L = self.lab_net(self.ph['label_input'], self.bit, self.dimLab)

            # construct image network
        self.Hsh_I, self.Fea_I_in, self.Fea_I_out= self.img_net(self.ph['image_input'], self.bit, self.numClass)

            # construct text network
        self.Hsh_T, self.Fea_T_in, self.Fea_T_out = self.txt_net(self.ph['text_input'], self.dimText, self.bit, self.numClass)

            # construct four discriminator networks
        self.isfrom_IL = self.dis_net_IL(self.ph['I_hash_batch'], self.ph['keep_prob'], reuse=False, name="disnet_IL")
        self.isfrom_L1 = self.dis_net_IL(self.ph['L_hash_batch'], self.ph['keep_prob'], reuse=True, name="disnet_IL")
        self.isfrom_TL = self.dis_net_TL(self.ph['T_hash_batch'], self.ph['keep_prob'], reuse=False, name="disnet_TL")
        self.isfrom_L2 = self.dis_net_TL(self.ph['L_hash_batch'], self.ph['keep_prob'], reuse=True, name="disnet_TL")

        self.isfrom_IAE_OUT = self.dis_net_IL(self.ph['IAE_OUT_batch'], self.ph['keep_prob'], reuse=False, name="disnet_IAE")
        self.isfrom_IAE_IN = self.dis_net_IL(self.ph['IAE_IN_batch'], self.ph['keep_prob'], reuse=True, name="disnet_IAE")
        self.isfrom_TAE_OUT = self.dis_net_TL(self.ph['TAE_OUT_batch'], self.ph['keep_prob'], reuse=False, name="disnet_TAE")
        self.isfrom_TAE_IN = self.dis_net_TL(self.ph['TAE_IN_batch'], self.ph['keep_prob'], reuse=True, name="disnet_TAE")

            # loss_D
        Loss_adver_IL = self.sce_loss(logits=self.isfrom_IL, labels=tf.zeros_like(self.isfrom_IL))
        Loss_adver_TL = self.sce_loss(logits=self.isfrom_TL, labels=tf.zeros_like(self.isfrom_TL))
        Loss_adver_L1 = self.sce_loss(logits=self.isfrom_L1, labels=tf.ones_like(self.isfrom_L1))
        Loss_adver_L2 = self.sce_loss(logits=self.isfrom_L2, labels=tf.ones_like(self.isfrom_L2))

        Loss_adver_IAE_OUT = self.sce_loss(logits=self.isfrom_IAE_OUT, labels=tf.zeros_like(self.isfrom_IAE_OUT))
        Loss_adver_TAE_OUT = self.sce_loss(logits=self.isfrom_TAE_OUT, labels=tf.zeros_like(self.isfrom_TAE_OUT))
        Loss_adver_IAE_IN = self.sce_loss(logits=self.isfrom_IAE_IN, labels=tf.ones_like(self.isfrom_IAE_IN))
        Loss_adver_TAE_IN = self.sce_loss(logits=self.isfrom_TAE_IN, labels=tf.ones_like(self.isfrom_TAE_IN))
        self.Loss_D = tf.div(Loss_adver_IL + Loss_adver_TL + Loss_adver_L1 + Loss_adver_L2 + \
                             Loss_adver_IAE_OUT + Loss_adver_TAE_OUT + Loss_adver_IAE_IN + Loss_adver_TAE_IN, 8.0)

            # train lab_net
        theta_L_1 = 1.0 / 2 * tf.matmul(self.ph['H'], tf.transpose(self.Hsh_L))
        Loss_pair_Hsh_L = self.mse_loss(tf.multiply(self.ph['Sim'], theta_L_1), tf.log(1.0 + tf.exp(theta_L_1)))
        Loss_quant_L = self.mse_loss(tf.squeeze(self.ph['L_hash_batch']), self.Hsh_L)
        Loss_classfication = self.mse_loss(self.ph['label_L'], self.Label_L)
        self.loss_l = gamma * Loss_pair_Hsh_L + beta * Loss_quant_L + eta * Loss_classfication

            # train img_net combined with lab_net
        theta_I_1 = 1.0 / 2 * tf.matmul(self.ph['H'], tf.transpose(self.Hsh_I))
        Loss_pair_Hsh_I = self.mse_loss(tf.multiply(self.ph['Sim'], theta_I_1), tf.log(1.0 + tf.exp(theta_I_1)))
        Loss_quant_I = self.mse_loss(tf.squeeze(self.ph['I_hash_batch']), self.Hsh_I)
        Loss_adver_I = self.sce_loss(logits=self.isfrom_IL, labels=tf.ones_like(self.isfrom_IL))
        Loss_adver_IAE = self.sce_loss(logits=self.isfrom_IAE_OUT, labels=tf.ones_like(self.isfrom_IAE_OUT))
        I_loss_pos = self.mse_loss(self.ph['T_pos'],tf.squeeze(self.ph['I_hash_batch']))
        I_loss_neg = self.mse_loss(self.ph['T_neg'],tf.squeeze(self.ph['I_hash_batch']))
        self.triplet_loss_I = tf.maximum(0.,0.1 + 5 * I_loss_pos - I_loss_neg)
        self.loss_i =gamma * Loss_pair_Hsh_I + beta * Loss_quant_I + delta * Loss_adver_I + \
                     alpha * Loss_adver_IAE + theta * self.triplet_loss_I

            # train txt_net combined with lab_net
        theta_T_1 = 1.0 / 2 * tf.matmul(self.ph['H'], tf.transpose(self.Hsh_T))
        Loss_pair_Hsh_T = self.mse_loss(tf.multiply(self.ph['Sim'], theta_T_1), tf.log(1.0 + tf.exp(theta_T_1)))
        Loss_quant_T = self.mse_loss(tf.squeeze(self.ph['T_hash_batch']), self.Hsh_T)
        Loss_adver_T = self.sce_loss(logits=self.isfrom_TL, labels=tf.ones_like(self.isfrom_TL))
        Loss_adver_TAE = self.sce_loss(logits=self.isfrom_TAE_OUT, labels=tf.ones_like(self.isfrom_TAE_OUT))
        T_loss_pos = self.mse_loss(self.ph['I_pos'],tf.squeeze(self.ph['T_hash_batch']))
        T_loss_neg = self.mse_loss(self.ph['I_neg'],tf.squeeze(self.ph['T_hash_batch']))
        self.triplet_loss_T = tf.maximum(0.,0.1 + 5 * T_loss_pos - T_loss_neg)
        self.loss_t = gamma * Loss_pair_Hsh_T + beta * Loss_quant_T + delta * Loss_adver_T + \
                      alpha * Loss_adver_TAE + theta * self.triplet_loss_T

    def train(self):
            # """Train"""
        dis_list = []
        gen_list = []
        for var in tf.trainable_variables():
            if var.name.startswith('discriminator'):
                dis_list.append(var)
            else:
                gen_list.append(var)

        optimizer = tf.train.AdamOptimizer(self.ph['lr_hash'])
        dis_optim = tf.train.AdamOptimizer(self.ph['lr_dis'])

        gradient_l = optimizer.compute_gradients(self.loss_l)
        self.train_lab = optimizer.apply_gradients(gradient_l)

        gradient_i = optimizer.compute_gradients(self.loss_i, var_list=gen_list)
        self.train_img = optimizer.apply_gradients(gradient_i)

        gradient_t = optimizer.compute_gradients(self.loss_t, var_list=gen_list)
        self.train_txt = optimizer.apply_gradients(gradient_t)

        gradient_D = dis_optim.compute_gradients(self.Loss_D, var_list=dis_list)
        self.train_dis = dis_optim.apply_gradients(gradient_D)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        var = {}
        var['lr_lab'] = self.lr_lab
        var['lr_dis'] = self.lr_dis
        var['lr_img'] = self.lr_img
        var['lr_txt'] = self.lr_txt

        var['batch_size'] = batch_size
        var['F'] = np.random.randn(self.num_train, self.bit)
        var['G'] = np.random.randn(self.num_train, self.bit)
        var['H'] = np.random.randn(self.num_train, self.bit)

        var['IAE_IN'] = np.random.randn(self.num_train, self.AE_FEA)
        var['IAE_OUT'] = np.random.randn(self.num_train, self.AE_FEA)
        var['TAE_IN'] = np.random.randn(self.num_train, self.AE_FEA)
        var['TAE_OUT'] = np.random.randn(self.num_train, self.AE_FEA)

        var['B'] = np.sign(var['H'] + var['G'] + var['F'])

        # Iterations
        for epoch in range(Epoch):
            results = {}
            results['loss_labNet'] = []
            results['loss_imgNet'] = []
            results['loss_txtNet'] = []
            results['Loss_D'] = []
            results['mapl2l'] = []
            results['mapi2i'] = []
            results['mapt2t'] = []

            if epoch % 1 == 0:
                print ('++++++++Start train lab_net++++++++')
                for idx in range(2):
                    lr_lab_Up = var['lr_lab'][epoch:]
                    lr_lab = lr_lab_Up[idx]
                    for train_labNet_k in range(int(k_lab_net/(idx+1))):
                        # Train lab_net
                        var['H'] = self.train_lab_net(var, lr_lab)
                        var['B'] = np.sign(var['H'])
                        train_labNet_loss = self.calc_labnet_loss(var['H'], Sim)
                        results['loss_labNet'].append(train_labNet_loss)
                        print ('---------------------------------------------------------------')
                        print ('...epoch: %3d, loss_labNet: %3.3f' % (epoch, train_labNet_loss))
                        print ('---------------------------------------------------------------')
                        if train_labNet_k > 1 and (results['loss_labNet'][-1] - results['loss_labNet'][-2]) >= 0:
                            break

                # Train domain discriminator
            if epoch % 1 == 0:
                print ('++++++++Start train dis_net++++++++')
                for idx in range(2):
                    lr_dis_Up = var['lr_dis'][epoch:]
                    lr_dis = lr_dis_Up[idx]
                    for train_disNet_k in range(k_dis_net):
                        IsFrom_, IsFrom, Loss_D = self.train_dis_net(lr_dis)
                        erro, acc = self.calc_isfrom_acc(IsFrom_, IsFrom)
                        results['Loss_D'].append(Loss_D)
                        print ('----------------------------------------')
                        print ('..epoch:{0}, Loss_D:{1}, acc:{2}'.format(epoch, Loss_D, acc))
                        print ('----------------------------------------')
                        if train_disNet_k > 1 and (results['Loss_D'][-1] - results['Loss_D'][-2]) <= 0:
                            break

            print ('++++++++Starting Train img_net++++++++')
            for idx in range(3):
                lr_img_Up = var['lr_img'][epoch:]
                lr_img = lr_img_Up[idx]
                for train_imgNet_k in range(int(k_img_net/(idx+1))):
                    # Train img_net
                    var['F'], var['IAE_IN'], var['IAE_OUT'], triplet_loss_i = self.train_img_net(var, lr_img)
                    B_i = np.sign(var['F'])
                    if train_imgNet_k % 2 == 0:
                        train_imgNet_loss = self.calc_loss(B_i, var['F'], var['H'], Sim, triplet_loss_i, gamma, beta)
                        results['loss_imgNet'].append(train_imgNet_loss)
                        print ('---------------------------------------------------------------')
                        print ('...epoch: %3d, loss_imgNet: %3.3f' % (epoch, train_imgNet_loss))
                        print ('---------------------------------------------------------------')
                    if train_imgNet_k > 2 and (results['loss_imgNet'][-1] - results['loss_imgNet'][-2]) >= 0:
                        break

            print ('++++++++Starting Train txt_net++++++++')
            for idx in range(3):
                lr_txt_Up = var['lr_txt'][epoch:]
                lr_txt = lr_txt_Up[idx]
                for train_txtNet_k in range(int(k_txt_net / (idx + 1))):
                    var['G'], var['TAE_IN'], var['TAE_OUT'], triplet_loss_t = self.train_txt_net(var, lr_txt)
                    B_t = np.sign(var['G'])
                    if train_txtNet_k % 2 == 0:
                        train_txtNet_loss = self.calc_loss(B_t, var['G'], var['H'], Sim, triplet_loss_t, gamma, beta)
                        results['loss_txtNet'].append(train_txtNet_loss)
                        print ('---------------------------------------------------------------')
                        print ('...epoch: %3d, Loss_txtNet: %s' % (epoch, train_txtNet_loss))
                        print ('---------------------------------------------------------------')
                    if train_txtNet_k > 2 and (results['loss_txtNet'][-1] - results['loss_txtNet'][-2]) >= 0:
                        break

            var['B'] = np.sign(var['H'] + var['G'] + var['F'])

            print ("********test************")
            self.test(self.phase)

            if np.mod(epoch, save_freq) == 0:
                self.save(self.checkpoint_dir, epoch)

    def test(self, phase):
        test = {}
        print ('==========================================================')
        print ('  ====                 Test map in all              ====')
        print ('==========================================================')

        if phase == 'test' and self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        test['qBX'] = self.generate_code(self.query_X, self.bit, "image")
        test['qBY'] = self.generate_code(self.query_Y, self.bit, "text")
        test['rBX'] = self.generate_code(self.retrieval_X, self.bit, "image")
        test['rBY'] = self.generate_code(self.retrieval_Y, self.bit, "text")

        test['mapi2t'] = calc_map(test['qBX'], test['rBY'], self.query_L, self.retrieval_L)
        test['mapt2i'] = calc_map(test['qBY'], test['rBX'], self.query_L, self.retrieval_L)
        test['mapi2i'] = calc_map(test['qBX'], test['rBX'], self.query_L, self.retrieval_L)
        test['mapt2t'] = calc_map(test['qBY'], test['rBY'], self.query_L, self.retrieval_L)
        print ('==================================================')
        print ('...test map: map(i->t): %3.3f, map(t->i): %3.3f' % (test['mapi2t'], test['mapt2i']))
        print ('...test map: map(t->t): %3.3f, map(i->i): %3.3f' % (test['mapt2t'], test['mapi2i']))
        print ('==================================================')

            # Save hash code
        datasetStr = DATA_DIR.split('/')[-1]
        datasetStr = datasetStr.split('.')[0]
        dataset_bit_net = datasetStr + str(bit) + netStr
        savePath = '/'.join([os.getcwd(), 'Savecode', dataset_bit_net + '.mat'])
        if os.path.exists(savePath):
            os.remove(savePath)
        sio.savemat(savePath, {'Qi': test['qBX'], 'Qt': test['qBY'],
                                      'Di': test['rBX'], 'Dt': test['rBY'],
                                      'retrieval_L': L['retrieval'], 'query_L': L['query']})

    def train_lab_net(self, var, lr_lab):
        print ('update label_net')
        H = var['H']
        batch_size = var['batch_size']
        num_train = self.train_L.shape[0]
        for iter in tqdm(range(int(num_train / batch_size))):
            index = np.random.permutation(num_train)
            ind = index[0: batch_size]
            sample_L = self.train_L[ind, :]
            label = self.train_L[ind, :].astype(np.float32)
            label = label.reshape([label.shape[0], 1, label.shape[1], 1])
            S = calc_neighbor(self.train_L, sample_L)
            result = self.sess.run([self.Hsh_L, self.Label_L], feed_dict={self.ph['label_input']: label})
            Hsh_L = result[0]
            Lab_L = result[1]

            H[ind, :] = Hsh_L
            self.train_lab.run(feed_dict={self.ph['Sim']: S,
                                          self.ph['H']: var['H'],
                                          self.ph['lr_hash']: lr_lab,
                                          self.ph['label_input']: label,
                                          self.ph['label_L']:Lab_L,
                                          self.ph['L_hash_batch']: np.sign(Hsh_L).reshape([Hsh_L.shape[0], 1, Hsh_L.shape[1], 1]),
                                          self.ph['keep_prob']: 1.0})
        return H

    def train_dis_net(self, lr):
        print ('update dis_net')
        for iter in range(int(num_train / batch_size)):
        # for iter in range(10):
            index = np.random.permutation(num_train)
            ind = index[0: batch_size]
            image = self.train_X[ind].astype(np.float64)
            image = image - self.meanpix.astype(np.float64)
            text = self.train_Y[ind, :].astype(np.float32)
            text = text.reshape([text.shape[0], 1, text.shape[1], 1])
            label = train_L[ind, :].astype(np.float32)
            label = label.reshape([label.shape[0], 1, label.shape[1], 1])
            result = self.sess.run([self.Hsh_I, self.Hsh_T, self.Hsh_L,self.Fea_I_in,
                                    self.Fea_I_out,self.Fea_T_in,self.Fea_T_out], feed_dict={self.ph['image_input']: image,
                                                                                    self.ph['text_input']: text,
                                                                                    self.ph['label_input']: label})
            Hsh_I = result[0]; Hsh_T = result[1];Hsh_L = result[2]
            Fea_I_in = result[3];Fea_I_out = result[4]
            Fea_T_in = result[5];Fea_T_out = result[6]

            self.train_dis.run(feed_dict={self.ph['L_hash_batch']: Hsh_L.reshape([Hsh_L.shape[0], 1, Hsh_L.shape[1], 1]),
                                          self.ph['I_hash_batch']: Hsh_I.reshape([Hsh_I.shape[0], 1, Hsh_I.shape[1], 1]),
                                          self.ph['T_hash_batch']: Hsh_T.reshape([Hsh_T.shape[0], 1, Hsh_T.shape[1], 1]),
                                          self.ph['IAE_OUT_batch']:Fea_I_out.reshape(
                                              [Fea_I_out.shape[0], 1, Fea_I_out.shape[1], 1]),
                                          self.ph['IAE_IN_batch']: Fea_I_in.reshape(
                                              [Fea_I_in.shape[0], 1, Fea_I_in.shape[1], 1]),
                                          self.ph['TAE_OUT_batch']: Fea_T_out.reshape(
                                              [Fea_T_out.shape[0], 1, Fea_T_out.shape[1], 1]),
                                          self.ph['TAE_IN_batch']: Fea_T_in.reshape(
                                              [Fea_T_in.shape[0], 1, Fea_T_in.shape[1], 1]),
                                          self.ph['lr_dis']: lr,
                                          self.ph['keep_prob']: 1.0})

            isfrom_IL = self.isfrom_IL.eval(feed_dict={self.ph['I_hash_batch']: Hsh_I.reshape([Hsh_I.shape[0], 1, Hsh_I.shape[1], 1]),
                                                   self.ph['keep_prob']: 1.0})
            isfrom_L1 = self.isfrom_L1.eval(feed_dict={self.ph['L_hash_batch']: Hsh_L.reshape([Hsh_L.shape[0], 1, Hsh_L.shape[1], 1]),
                                                   self.ph['keep_prob']: 1.0})
            isfrom_TL = self.isfrom_TL.eval(feed_dict={self.ph['T_hash_batch']: Hsh_T.reshape([Hsh_T.shape[0], 1, Hsh_T.shape[1], 1]),
                                                   self.ph['keep_prob']: 1.0})


            Loss_Dis = self.Loss_D.eval(feed_dict={self.ph['L_hash_batch']: Hsh_L.reshape([Hsh_L.shape[0], 1, Hsh_L.shape[1], 1]),
                                          self.ph['I_hash_batch']: Hsh_I.reshape([Hsh_I.shape[0], 1, Hsh_I.shape[1], 1]),
                                          self.ph['T_hash_batch']: Hsh_T.reshape([Hsh_T.shape[0], 1, Hsh_T.shape[1], 1]),
                                          self.ph['IAE_OUT_batch']:Fea_I_out.reshape(
                                              [Fea_I_out.shape[0], 1, Fea_I_out.shape[1], 1]),
                                          self.ph['IAE_IN_batch']: Fea_I_in.reshape(
                                              [Fea_I_in.shape[0], 1, Fea_I_in.shape[1], 1]),
                                          self.ph['TAE_OUT_batch']: Fea_T_out.reshape(
                                              [Fea_T_out.shape[0], 1, Fea_T_out.shape[1], 1]),
                                          self.ph['TAE_IN_batch']: Fea_T_in.reshape(
                                              [Fea_T_in.shape[0], 1, Fea_T_in.shape[1], 1]),
                                          self.ph['lr_dis']: lr,
                                          self.ph['keep_prob']: 1.0})
            if iter % 5 == 0:
                print ('...discriminator_Loss_D: {0}'.format(Loss_Dis))
        return np.hstack((isfrom_IL, isfrom_L1, isfrom_TL)), np.hstack((np.zeros_like(isfrom_IL), np.ones_like(isfrom_L1), np.zeros_like(isfrom_TL))), Loss_Dis

    
    def train_img_net(self, var, lr_img):
    
        print ('update image_net')
        F = var['F']
        IAE_IN = var['IAE_IN']
        IAE_OUT = var['IAE_OUT']
        batch_size = var['batch_size']
        num_train = self.train_X.shape[0]
        for iter in tqdm(range(int(num_train / batch_size))):
        # for iter in tqdm(range(5)):
            index = np.random.permutation(num_train)
            ind = index[0: batch_size]
            sample_L = train_L[ind, :]
            label = train_L[ind, :].astype(np.float32)
            calc_sim = -1 * dot(label, label.T)
            index_neg = argmax(calc_sim, axis=0)
            label = label.reshape([label.shape[0], 1, label.shape[1], 1])
            image = self.train_X[ind, :, :, :].astype(np.float64)
            image = image - self.meanpix.astype(np.float64)
            text = self.train_Y[ind, :].astype(np.float32)
            text = text.reshape([text.shape[0], 1, text.shape[1], 1])
            S = calc_neighbor(train_L, sample_L)
            result = self.sess.run([self.Hsh_I, self.Fea_I_in, self.Fea_I_out, self.Hsh_T], \
                                   feed_dict={self.ph['image_input']: image, self.ph['text_input']: text})

            Hsh_I = result[0]
            Fea_I_in = result[1]
            Fea_I_out = result[2]
            Hsh_T = result[3]

            F[ind, :] = Hsh_I
            IAE_IN[ind, :] = Fea_I_in
            IAE_OUT[ind, :] = Fea_I_out

            pos = Hsh_T
            neg = Hsh_T[index_neg, :]

            self.train_img.run(feed_dict={self.ph['Sim']: S,
                                        self.ph['H']: var['H'],
                                        self.ph['I_hash_batch']: np.sign(Hsh_I).reshape([Hsh_I.shape[0], 1, Hsh_I.shape[1], 1]),
                                        self.ph['T_pos']:pos, self.ph['T_neg']:neg,
                                        self.ph['IAE_OUT_batch']: Fea_I_out.reshape([Fea_I_out.shape[0], 1, Fea_I_out.shape[1], 1]),
                                        self.ph['lr_hash']: lr_img,
                                        self.ph['image_input']: image,
                                        self.ph['label_input']: label,
                                        self.ph['keep_prob']: 1.0})

            triplet_loss_i = self.triplet_loss_I.eval(feed_dict={self.ph['I_hash_batch']: np.sign(Hsh_I).reshape([Hsh_I.shape[0], 1, Hsh_I.shape[1], 1]),
                                        self.ph['T_pos']:pos, self.ph['T_neg']:neg})
        return F, IAE_IN,IAE_OUT, triplet_loss_i
    
    
    def train_txt_net(self, var, lr_txt):
        print ('update text_net')
    
        G = var['G']
        TAE_IN = var['TAE_IN']
        TAE_OUT = var['TAE_OUT']
        batch_size = var['batch_size']
        num_train = self.train_Y.shape[0]
        for iter in tqdm(range(int(num_train / batch_size))):
        # for iter in tqdm(range(5)):
            index = np.random.permutation(num_train)
            ind = index[0: batch_size]
            sample_L = train_L[ind, :]
            label = train_L[ind, :].astype(np.float32)
            calc_sim = -1 * dot(label, label.T)
            index_neg = argmax(calc_sim, axis=0)
            label = label.reshape([label.shape[0], 1, label.shape[1], 1])
            image = self.train_X[ind, :, :, :].astype(np.float64)
            image = image - self.meanpix.astype(np.float64)
            text = self.train_Y[ind, :].astype(np.float32)
            text = text.reshape([text.shape[0], 1, text.shape[1], 1])

            S = calc_neighbor(train_L, sample_L)
            result = self.sess.run([self.Hsh_T, self.Fea_T_in, self.Fea_T_out, self.Hsh_I], \
                                   feed_dict={self.ph['image_input']: image, self.ph['text_input']: text})

            Hsh_T = result[0]
            Fea_T_in = result[1]
            Fea_T_out = result[2]
            Hsh_I = result[3]

            G[ind, :] = Hsh_T
            TAE_IN[ind, :] = Fea_T_in
            TAE_OUT[ind, :] = Fea_T_out

            pos = Hsh_I
            neg = Hsh_I[index_neg, :]

            self.train_txt.run(feed_dict={self.ph['Sim']: S,
                                          self.ph['H']: var['H'],
                                          self.ph['T_hash_batch']: np.sign(Hsh_T).reshape([Hsh_T.shape[0], 1, Hsh_T.shape[1], 1]),
                                          self.ph['I_pos']: pos, self.ph['I_neg']: neg,
                                          self.ph['TAE_OUT_batch']: Fea_T_out.reshape([Fea_T_out.shape[0], 1, Fea_T_out.shape[1], 1]),
                                          self.ph['lr_hash']: lr_txt,
                                          self.ph['label_input']: label,
                                          self.ph['text_input']: text,
                                          self.ph['keep_prob']: 1.0})
            triplet_loss_t = self.triplet_loss_T.eval(feed_dict={self.ph['T_hash_batch']: np.sign(Hsh_T).reshape([Hsh_T.shape[0], 1, Hsh_T.shape[1], 1]),
                           self.ph['I_pos']: pos, self.ph['I_neg']: neg})
        return G, TAE_IN, TAE_OUT, triplet_loss_t

    def generate_code(self, Modal, bit, generate):
        batch_size = 128
        if generate=="label":
            num_data = Modal.shape[0]
            index = np.linspace(0, num_data - 1, num_data).astype(int)
            B = np.zeros([num_data, bit], dtype=np.float32)
            for iter in tqdm(range(int(num_data / batch_size + 1))):
                ind = index[iter * batch_size: min((iter + 1) * batch_size, num_data)]
                label = Modal[ind, :].astype(np.float32)
                label = label.reshape([label.shape[0], 1, label.shape[1], 1])
                Hsh_L = self.Hsh_L.eval(feed_dict={self.ph['label_input']: label})
                B[ind, :] = Hsh_L
        elif generate=="image":
            num_data = len(Modal)
            index = np.linspace(0, num_data - 1, num_data).astype(int)
            B = np.zeros([num_data, bit], dtype=np.float32)
            for iter in tqdm(range(int(num_data / batch_size + 1))):
                ind = index[iter * batch_size: min((iter + 1) * batch_size, num_data)]
                mean_pixel = np.repeat(self.meanpix[:, :, :, np.newaxis], len(ind), axis=3)
                image = Modal[ind,:,:,:].astype(np.float64)
                image = image - mean_pixel.astype(np.float64).transpose(3, 0, 1, 2)
                Hsh_I = self.Hsh_I.eval(feed_dict={self.ph['image_input']: image})
                B[ind, :] = Hsh_I
        else:
            num_data = Modal.shape[0]
            index = np.linspace(0, num_data - 1, num_data).astype(int)
            B = np.zeros([num_data, bit], dtype=np.float32)
            for iter in tqdm(range(int(num_data / batch_size + 1))):
                ind = index[iter * batch_size: min((iter + 1) * batch_size, num_data)]
                text = Modal[ind, :].astype(np.float32)
                text = text.reshape([text.shape[0], 1, text.shape[1], 1])
                Hsh_T = self.Hsh_T.eval(feed_dict={self.ph['text_input']: text})
                B[ind, :] = Hsh_T
        B = np.sign(B)
        return B

    def calc_labnet_loss(self, H, SIM):

        theta = np.matmul(H, np.transpose(H)) / 2
        term = np.sum(np.log(1 + np.exp(theta)) - SIM * theta)

    
        loss = gamma * term
        print ('pairwise_hash:',term)
        return loss
    
    
    def calc_loss(self, B, F, H, Sim, triplet_loss, gamma, beta):
        theta = np.matmul(F, np.transpose(H)) / 2
        term1 = np.sum(np.log(1 + np.exp(theta)) - Sim * theta)

        term2 = np.sum(np.power(B-F, 2))
    
        loss = gamma * term1 + beta * term2 + triplet_loss
        print ('pairwise:', term1)
        print ('quantization:', term2)
        print('triplet loss:', triplet_loss)
        return loss


    def calc_isfrom_acc(self, train_isfrom_, Train_ISFROM):
        erro = Train_ISFROM.shape[0] - np.sum(np.equal(np.sign(train_isfrom_ - 0.5), np.sign(Train_ISFROM - 0.5)).astype(int))
        acc = np.divide(np.sum(np.equal(np.sign(train_isfrom_ - 0.5), np.sign(Train_ISFROM - 0.5)).astype('float32')), Train_ISFROM.shape[0])
        return erro, acc


    def save(self, checkpoint_dir, step):
        model_name = "adversial_AE_hashing"
        model_dir = "%s_%s" % (self.dataset_dir, self.bit)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.dataset_dir, self.bit)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False