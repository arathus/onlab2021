class Trade:
	def __init__(self, instrument, type, units, row, stop_loss, take_profit, direct):
		self.Type = type								# BUY or SELL
		self.Units = units								# trade ammount
		self.SL = stop_loss								# stop loss
		self.TP = take_profit							# take Profit
		self.OT = row['Date'] + ' ' + row['Time']		# open time
		self.CT = None		 							# close time
		self.OP = 0 									# open price
		self.CP = 0										# close price
		self.Profit = 0									# calculated Profit for pessimistic TP
		self.Instrument = instrument.upper()			# instrument
		self.Closed = False 							# to check if the trade is closed
		self.AutoClose = False							# checks if closed automatically
		self.Direct = direct 							# base currency is also account currency
		
		# set opening price based on order Type
		if self.Type == 'BUY':
			self.OP = row['AC']
		else:
			self.OP = row['BC']
	

	def __str__(self):
	 return self.Type + ' ' + str(self.Units) + ' ' + self.OT

	def asdict(self):
		return {'Type': self.Tpye,
				'Units': self.Units,
				'SL': self.SL,
				'TP': self.TP, 
				'OT': self.OT,
				'CT': self.CT,
				'OP': self.OP,
				'CP': self.CP,
				'Profit': self.Profit,
				'Instrument': self.Instrument,
				'Closed': self.Closed }
	

	def update(self, row):
		# Pessimistic SL/TP implementation:
		if self.Direct:
			loss = 0
			if self.Type == 'BUY':
				self.Profit = (row['BH'] - self.OP) * (1 / row['BH']) * self.Units
				loss = (row['BL'] - self.OP) * (1 / row['BL']) * self.Units
			elif self.Type == 'SELL':
				self.Profit =  (self.OP - row['AL']) * (1 / row['AL']) * self.Units
				loss = (self.OP - row['AH']) * (1 / row['AH']) * self.Units
			
			# If we had a stop loss at the lowest price it means
			# that we have lost our SL money at some point.
			if self.SL != 0 and loss <= -self.SL:	
				self.Profit = -self.SL
				self.AutoClose = True
				self.close(row)
				return True

			# If we had take profit triggered that means that SL did not occur during this 
			# period so we've won the initially given TP money.
			if self.TP != 0 and self.Profit >= self.TP:
				self.Profit = self.TP
				self.AutoClose = True
				self.close(row)
				return True

			# Recalculate profits on close price so it is closeable correctly by a strategy.
			if self.Type == 'BUY':
				self.Profit = (row['BC'] - self.OP) * (1 / row['BC']) * self.Units
			elif self.Type == 'SELL':
				self.Profit =  (self.OP - row['AC']) * (1 / row['AC']) * self.Units
		else:
			loss = 0
			if self.Type == 'BUY':
				self.Profit = (row['BH'] - self.OP) * self.Units
				loss = (row['BL'] - self.OP) * self.Units
			elif self.Type == 'SELL':
				self.Profit =  (self.OP - row['AL']) * self.Units
				loss = (self.OP - row['AH']) * self.Units
			
			# If we had a stop loss at the lowest price it means
			# that we have lost our SL money at some point.
			if self.SL != 0 and loss <= -self.SL:	
				self.Profit = -self.SL
				self.AutoClose = True
				self.close(row)
				return True

			# If we had take profit triggered that means that SL did not occur during this 
			# period so we've won the initially given TP money.
			if self.TP != 0 and self.Profit >= self.TP:
				self.Profit = self.TP
				self.AutoClose = True
				self.close(row)
				return True

			# Recalculate profits on close price so it is closeable correctly by a strategy.
			if self.Type == 'BUY':
				self.Profit = (row['BC'] - self.OP) * self.Units
			elif self.Type == 'SELL':
				self.Profit =  (self.OP - row['AC']) * self.Units
		
		return False
	

	def close(self, row):
		if not self.Closed:
			self.CT = row['Date'] + ' ' + row['Time']	
			if self.Type == 'BUY':
				if self.Profit > 0:
					if self.AutoClose:
						self.CP = row['BH']
					else:
						self.CP = row['BC']
				else:
					if self.AutoClose:
						self.CP = row['BL']
					else:
						self.CP = row['BC']
			elif self.Type == 'SELL':
				if self.Profit > 0:
					if self.AutoClose:
						self.CP = row['AL']
					else:
						self.CP = row['AC']
				else:
					if self.AutoClose:
						self.CP = row['AH']
					else:
						self.CP = row['AC']
		self.Closed = True 
		