import copy, sys, re, ast, json
template_dict = {
"Request Meeting": {"Meeting Members": None, "Meeting Agenda": None, "Meeting Name":None, "Meeting Location":None,  "Meeting Date":None, "Meeting Time":None},
"Request Data": {'Context: Request Date':None, 'Context: Data idString':None, 'Context: Request Time':None, 'Context: Request members':None, 'Context: Data Owner':None, 'Context: Data Type':None},
"Request Action": {'Action Date':None, 'Action Members':None, 'Action Description':None, 'Action Time':None},
"Request Action Data": {'Context: Action Time': None, 'Context: Action Members': None, 'Context: Action Description': None, 'Context: Request Members': None, 'Context: Action Date': None},
"Request Meeting Data": {'Context: Meeting Date': None, 'Context: Meeting Agenda': None, 'Context: Meeting Time': None, 'Context: Meeting Location': None, 'Context: Meeting Members': None, 'Context: Request Members': None, 'Context: Meeting Name': None},
"Deliver Data": {'Deliver Members': None, 'Data Value': None, 'Deliver Date': None, 'Data idString': None, 'Deliver Time': None, 'Data Type': None},
"Deliver Action Data": {'Action Date': None, 'Action Members': None, 'Action Description': None, 'Action Time': None},
"Deliver Meeting Data": {'Meeting Members': None, 'Meeting Name': None, 'Meeting Agenda': None, 'Meeting Time': None, 'Meeting Date': None, 'Meeting Location': None},
"Amend Data": {'Context: Data Type': None, 'Revision: Data Type': None, 'Context: Data Value': None, 'Revision: Data Value': None, "Context: Amend Date": None, "Context: Amend Time": None, "Context: Amend Members": None},
"Amend Meeting Data":{"Context: Meeting Members": None, "Revision: Meeting Members": None, "Context: Meeting Agenda": None, "Revision: Meeting Agenda": None, "Context: Meeting Name": None, "Context: Meeting Location": None, "Revision: Meeting Location": None, "Context: Meeting Date": None, "Revision: Meeting Date": None, "Context: Meeting Time": None, "Revision: Meeting Time": None, "Context: Amend Date": None, "Revision: Amend Date": None, "Context: Amend Time": None, "Revision: Amend Time": None, "Context: Amend Members": None,"Revision: Amend Members": None}
}
class event_class:
    stats_dict = {}
    def __init__(self, event_name, templateDict, realDict, trigger, par = True):
        self.event_dict = templateDict
        self.par = par
        self.event_name = event_name
        self.trigger = trigger
        self.masked_templates = None
        for args in realDict:
            if(args=="trigger"): continue
            try:
                assert args!="trigger" and args in templateDict.keys() is not None
                self.event_dict[args] = realDict[args]
            except:
                raise AssertionError('Arg >{}< is not defined in the template for the event {}'.format(args, self.event_name))
    def prep_args(self, ref_dict = None):
        if(self.par==True):
            parBeg = ' | '
            parEnd = ' |'
        else:
            parBeg = parEnd = ""
        for argn in self.event_dict:
            argv = self.event_dict[argn]
            argv = "{}{}{}".format(parBeg, argn if (argv is None or len(argv)==0) else argv , parEnd) #if (argv is None or len(argv)==0)else argv
            self.event_dict[argn] = argv
        self.rev_event_dict = dict([(value, key) for key, value in self.event_dict.items()])
        self.trigger_template = f"Event {self.event_name} is triggered by | {self.trigger} | where , "
    def get_masked_template(self):
        assert self.masked_templates is not None
        return ""#self.trigger_template + self.masked_templates[0][0].strip() #+ "."
    def get_filled_template(self):
        return ""+self.trigger_template + self.fill_template().strip() #+ "."
    def clean_(self, this, withthis, inthis):
        return inthis.replace(this, withthis)
    def extract_arguments(self):
        #we don't use it anymore
        return "" # template_mappings
        

class request_meeting(event_class):
    def __init__(self, eventDict, garbage, trigger):
        super().__init__("Request Meeting", {"Meeting Members": None, "Meeting Agenda": None, "Meeting Name":None, "Meeting Location":None,  "Meeting Date":None, "Meeting Time":None}, eventDict, trigger)
        self.prep_args()
        self.masked_templates = [("{} is requested among {} at {} on {} at {} to discuss {}", None)]
    def fill_template(self):
        return '{} is requested among {} at {} on {} at {} to discuss {}'.format(self.event_dict["Meeting Name"] ,self.event_dict["Meeting Members"], self.event_dict["Meeting Time"], self.event_dict["Meeting Date"], self.event_dict["Meeting Location"], self.event_dict["Meeting Agenda"])


#rm = request_meeting({"Meeting Members": None, "Meeting Agenda": "", "Meeting Name":"Board Meeting", "Meeting Location":None,  "Meeting Date":"Tuesday"})


class request_data(event_class):
    def __init__(self, eventDict, req_attributes, trigger):
        super().__init__("Request Data", {"Data Value":None, 'Context: Request Date':None, 'Context: Data idString':None, 'Context: Request Time':None, 'Context: Request members':None, 'Context: Data Owner':None, 'Context: Data Type':None}, eventDict, trigger)
        self.prep_args()
        self.reqAtt =req_attributes
        self.masked_templates = [('{} of {} by {} is requested from {} to be delivered at {} on {}', 'Data Value'), ('Owner of {} of {} is requested from {} to be delivered at {} on {}', 'Data Owner')]
    def fill_template(self):
        if( event_class.stats_dict.get("Request_Data: " + self.reqAtt) is None):
            event_class.stats_dict["Request_Data: " + self.reqAtt] = 0
        event_class.stats_dict["Request_Data: " + self.reqAtt] += 1
        if(self.reqAtt in ["Data Value", ""]):
            return '{} of {} by {} is requested from {} to be delivered at {} on {}'.format(self.event_dict["Context: Data idString"] ,self.event_dict["Context: Data Type"], self.event_dict["Context: Data Owner"], self.event_dict["Context: Request members"], self.event_dict["Context: Request Time"], self.event_dict["Context: Request Date"])
        elif(self.reqAtt=="Data Owner"):
            return 'Owner of {} of {} is requested from {} to be delivered at {} on {}'.format(self.event_dict["Context: Data idString"] ,self.event_dict["Context: Data Type"], self.event_dict["Context: Request members"], self.event_dict["Context: Request Time"], self.event_dict["Context: Request Date"])
        else:
            raise AttributeError('Request Attribute {} is not supported for Request Data event'.format(self.reqAtt))

class request_action(event_class):
    def __init__(self, eventDict, garbage, trigger):
        super().__init__("Request Action", {'Action Date':None, 'Action Members':None, 'Action Description':None, 'Action Time':None}, eventDict, trigger)
        self.prep_args()
        self.masked_templates = [('{} is requested from {} at {} on {}', None)]
    def fill_template(self):
        return '{} is requested from {} at {} on {}'.format(self.event_dict["Action Description"] ,self.event_dict["Action Members"], self.event_dict["Action Time"], self.event_dict["Action Date"])

# ra1 = request_action({'Action Date':'Tuesday', 'Action Description':"sign the report", 'Action Time':"3 am", 'Action Members':"[a, b]"})

#Request Members
#Request Time
# 
class request_action_data(event_class):
    def __init__(self, eventDict, req_attributes, trigger):
        super().__init__("Request Action Data", {'Context: Action Time': None, 'Context: Action Members': None, 'Context: Action Description': None, 'Context: Request Members': None, 'Context: Action Date': None}, eventDict, trigger)
        self.prep_args()
        self.reqAtt =req_attributes
        self.masked_templates = [('Action Members is requested for {} at {} on {} from {}', "Action Members"), ('Date is requested for {} by {} at {} from {}', "Action Date"), ('Time is requested for {} by {} on {} from {}', "Action Time"), ("Action Description is requested for {} by {} at {} from {}", "Action Description")]
    def fill_template(self):
        if( event_class.stats_dict.get("Request_Action_Data: " + self.reqAtt) is None):
            event_class.stats_dict["Request_Action_Data: " + self.reqAtt] = 0
        event_class.stats_dict["Request_Action_Data: " + self.reqAtt] += 1
        if(self.reqAtt in ["Action Members", "Request Members"]):
            return 'Action Members is requested for {} at {} on {} from {}'.format(self.event_dict["Context: Action Description"] ,self.event_dict["Context: Action Time"], self.event_dict["Context: Action Date"], self.event_dict["Context: Request Members"])
        elif(self.reqAtt=="Action Date"):
            return 'Date is requested for {} by {} at {} from {}'.format(self.event_dict["Context: Action Description"] ,self.event_dict["Context: Action Members"], self.event_dict["Context: Action Time"], self.event_dict["Context: Request Members"])
        elif(self.reqAtt in ["Action Time", "Request Time"]):
            return 'Time is requested for {} by {} on {} from {}'.format(self.event_dict["Context: Action Description"] ,self.event_dict["Context: Action Members"], self.event_dict["Context: Action Date"], self.event_dict["Context: Request Members"])
        elif(self.reqAtt == "Action Description"):
            return "Action Description is requested for {} by {} at {} from {}".format(self.event_dict["Context: Action Description"], self.event_dict["Context: Action Members"], self.event_dict["Context: Action Time"], self.event_dict["Context: Request Members"])
        else:
            raise AttributeError('Request Attribute {} is not supported for Request Action Data event'.format(self.reqAtt))

# rad1 = request_action_data({'Context: Action Time': None, 'Context: Action Members': None, 'Context: Action Description': None, 'Context: Request Members': None, 'Context: Action Date': None}, 'Context: Action Time')


class request_meeting_data(event_class):
    def __init__(self, eventDict, req_attributes, trigger):
        super().__init__("Request Meeting Data", {'Context: Meeting Date': None, 'Context: Meeting Agenda': None, 'Context: Meeting Time': None, 'Context: Meeting Location': None, 'Context: Meeting Members': None, 'Context: Request Members': None, 'Context: Meeting Name': None}, eventDict, trigger)
        self.prep_args()
        self.reqAtt =req_attributes
        self.masked_templates = [('Meeting Members is requested for {} at {} on {} at {} to discuss {} from {}', "Meeting Members"), ('Date is requested for {} among {} at {} at {} to discuss {} from {}', "Meeting Date"), ('Time is requested for {} among {} on {} at {} to discuss {} from {}', "Meeting Time"), ('Location is requested for {} among {} at {} on {} to discuss {} from {}', "Meeting Location"), ('Agenda is requested for {} among {} at {} on {} at {} from {}', "Meeting Agenda")]
    def fill_template(self):
        if( event_class.stats_dict.get("Request_Meeting_Data: " + self.reqAtt) is None):
             event_class.stats_dict["Request_Meeting_Data: " + self.reqAtt] = 0
        event_class.stats_dict["Request_Meeting_Data: " + self.reqAtt] += 1
        if(self.reqAtt=="Meeting Members"):
            return 'Meeting Members is requested for {} at {} on {} at {} to discuss {} from {}'.format(self.event_dict["Context: Meeting Name"] ,self.event_dict["Context: Meeting Time"], self.event_dict["Context: Meeting Date"], self.event_dict["Context: Meeting Location"], self.event_dict["Context: Meeting Agenda"], self.event_dict["Context: Request Members"])
        elif(self.reqAtt=="Meeting Date"):
            return 'Date is requested for {} among {} at {} at {} to discuss {} from {}'.format(self.event_dict["Context: Meeting Name"], self.event_dict["Context: Meeting Members"], self.event_dict["Context: Meeting Time"], self.event_dict["Context: Meeting Location"], self.event_dict["Context: Meeting Agenda"], self.event_dict["Context: Request Members"])
        elif(self.reqAtt=="Meeting Time"):
            return 'Time is requested for {} among {} on {} at {} to discuss {} from {}'.format(self.event_dict["Context: Meeting Name"], self.event_dict["Context: Meeting Members"], self.event_dict["Context: Meeting Date"], self.event_dict["Context: Meeting Location"], self.event_dict["Context: Meeting Agenda"], self.event_dict["Context: Request Members"])
        elif(self.reqAtt=="Meeting Location"):
            return 'Location is requested for {} among {} at {} on {} to discuss {} from {}'.format(self.event_dict["Context: Meeting Name"], self.event_dict["Context: Meeting Members"], self.event_dict["Context: Meeting Time"], self.event_dict["Context: Meeting Date"], self.event_dict["Context: Meeting Agenda"], self.event_dict["Context: Request Members"])
        elif(self.reqAtt=="Meeting Agenda"):
            return 'Agenda is requested for {} among {} at {} on {} at {} from {}'.format(self.event_dict["Context: Meeting Name"], self.event_dict["Context: Meeting Members"], self.event_dict["Context: Meeting Time"], self.event_dict["Context: Meeting Date"], self.event_dict["Context: Meeting Location"], self.event_dict["Context: Request Members"])
        else:
            raise AttributeError('Request Attribute {} is not supported for Request Meeting Data event'.format(self.reqAtt))


class deliver_data(event_class):
    def __init__(self, eventDict, deliver_action, trigger):
        super().__init__("Deliver Data", {'Deliver Members': None, 'Data Value': None, 'Deliver Date': None, 'Data idString': None, 'Deliver Time': None, 'Data Type': None}, eventDict, trigger)
        self.prep_args()
        self.deliver_action = deliver_action if deliver_action!="" else "Positive"
        assert self.deliver_action in ["Positive", "Negative", "Unsure"]
        self.masked_templates = [("{}, {} of {} is or will be delivered to {} at {} on {}", "Positive"), ("{}, {} of {} is not or will not be delivered to {} at {} on {}", "Negative"), ("{}, {} of {} is or will probably be delivered to {} at {} on {}", "Unsure")]
    def fill_template(self):
        if( event_class.stats_dict.get("Deliver_Data: " + self.deliver_action) is None):
            event_class.stats_dict["Deliver_Data: " + self.deliver_action] = 0
        event_class.stats_dict["Deliver_Data: " + self.deliver_action] += 1
        if(self.deliver_action=="Positive"):
            return "{}, {} of {} is or will be delivered to {} at {} on {}".format(self.event_dict["Data idString"], self.event_dict["Data Value"],self.event_dict["Data Type"], self.event_dict["Deliver Members"], self.event_dict["Deliver Time"], self.event_dict["Deliver Date"])
        elif(self.deliver_action=="Negative"):
            return "{}, {} of {} is not or will not be delivered to {} at {} on {}".format(self.event_dict["Data idString"], self.event_dict["Data Value"],self.event_dict["Data Type"], self.event_dict["Deliver Members"], self.event_dict["Deliver Time"], self.event_dict["Deliver Date"] )
        elif(self.deliver_action=="Unsure"):
            return "{}, {} of {} is or will probably be delivered to {} at {} on {}".format(self.event_dict["Data idString"], self.event_dict["Data Value"],self.event_dict["Data Type"], self.event_dict["Deliver Members"], self.event_dict["Deliver Time"], self.event_dict["Deliver Date"] )
        else:
            raise AttributeError('Deliver Action {} is not supported for Deliver Data event'.format(self.deliver_action))





class deliver_action_data(event_class):
    def __init__(self, eventDict, deliver_action, trigger):
        super().__init__("Deliver Action Data", {'Action Date': None, 'Action Members': None, 'Action Description': None, 'Action Time': None}, eventDict, trigger)
        self.prep_args()
        self.deliver_action = deliver_action if deliver_action!="" else "Positive"
        assert self.deliver_action in ["Positive", "Negative", "Unsure"]
        self.masked_templates = [("{} is or will be performed by {} at {} on {}", "Positive"), ("{} is not or will not be performed by {} at {} on {}", "Negative"), ("{} is or will probably be performed by {} at {} on {}", "Unsure")]
    def fill_template(self):
        if( event_class.stats_dict.get("Deliver_Action_Data: " + self.deliver_action) is None):
             event_class.stats_dict["Deliver_Action_Data: " + self.deliver_action] = 0
        event_class.stats_dict["Deliver_Action_Data: " + self.deliver_action] += 1
        if(self.deliver_action=="Positive"):
            return "{} is or will be performed by {} at {} on {}".format(self.event_dict["Action Description"], self.event_dict["Action Members"],self.event_dict["Action Time"], self.event_dict["Action Date"])
        elif(self.deliver_action=="Negative"):
            return "{} is not or will not be performed by {} at {} on {}".format(self.event_dict["Action Description"], self.event_dict["Action Members"],self.event_dict["Action Time"], self.event_dict["Action Date"])
        elif(self.deliver_action=="Unsure"):
            return "{} is or will probably be performed by {} at {} on {}".format(self.event_dict["Action Description"], self.event_dict["Action Members"],self.event_dict["Action Time"], self.event_dict["Action Date"])
        else:
            raise AttributeError('Deliver Action {} is not supported for Deliver Action Data event'.format(self.deliver_action))





class deliver_meeting_data(event_class):
    def __init__(self, eventDict, deliver_action, trigger):
        super().__init__("Deliver Meeting Data", {'Meeting Members': None, 'Meeting Name': None, 'Meeting Agenda': None, 'Meeting Time': None, 'Meeting Date': None, 'Meeting Location': None}, eventDict, trigger)
        self.prep_args()
        self.deliver_action = deliver_action if deliver_action!="" else "Unsure"
        assert self.deliver_action in ["Positive", "Negative", "Unsure"]    
        self.masked_templates = [("{} is or will be attended by {} at {} on {} at {} to discuss {}", "Positive"), ("{} is not or will not be attended by {} at {} on {} at {} to discuss {}", "Negative"), ("{} is or will probably be attended by {} at {} on {} at {} to discuss {}", "Unsure")]    
    def fill_template(self):
        if( event_class.stats_dict.get("Deliver_Meeting_Data: " + self.deliver_action) is None):
             event_class.stats_dict["Deliver_Meeting_Data: " + self.deliver_action] = 0
        event_class.stats_dict["Deliver_Meeting_Data: " + self.deliver_action] += 1
        if(self.deliver_action=="Positive"):
            return "{} is or will be attended by {} at {} on {} at {} to discuss {}".format(self.event_dict["Meeting Name"], self.event_dict["Meeting Members"], self.event_dict["Meeting Time"], self.event_dict["Meeting Date"], self.event_dict["Meeting Location"], self.event_dict["Meeting Agenda"])
        elif(self.deliver_action=="Negative"):
            return "{} is not or will not be attended by {} at {} on {} at {} to discuss {}".format(self.event_dict["Meeting Name"], self.event_dict["Meeting Members"], self.event_dict["Meeting Time"], self.event_dict["Meeting Date"], self.event_dict["Meeting Location"], self.event_dict["Meeting Agenda"])
        elif(self.deliver_action=="Unsure"):
            return "{} is or will probably be attended by {} at {} on {} at {} to discuss {}".format(self.event_dict["Meeting Name"], self.event_dict["Meeting Members"], self.event_dict["Meeting Time"], self.event_dict["Meeting Date"], self.event_dict["Meeting Location"], self.event_dict["Meeting Agenda"])
        else:
            raise AttributeError('Deliver Action {} is not supported for Deliver Meeting Data event'.format(self.deliver_action))

class amend_data(event_class):
    def __init__(self, eventDict, amend_action, trigger):
        super().__init__("Amend Data", {'Context: Data Type': None, 'Revision: Data Type': None, 'Context: Data Value': None, 'Revision: Data Value': None, "Context: Amend Date": None, "Context: Amend Time": None, "Context: Amend Members": None, "Context: Data idString":None, "Revision: Data idString":None}, eventDict, trigger)
        self.prep_args()
        self.amend_action = amend_action
        assert self.amend_action in ["Update", "Add", "Delete", ""] and amend_action.strip() is not None
        self.masked_templates = [("For {}, {} is or requested to be updated to {} from {} at {} on {}", "Update"),( "For {}, {} is or requested to be added from {} at {} on {}", "Add"), ("For {}, {} is or requested to be removed from {} at {} on {}", "Delete")]        
    def fill_template(self):
        if( event_class.stats_dict.get("Amend_Data: " + self.amend_action) is None):
             event_class.stats_dict["Amend_Data: " + self.amend_action] = 0
        event_class.stats_dict["Amend_Data: " + self.amend_action] += 1
        if(self.amend_action in ["Update", ""]):
            return "For {}, {} is or requested to be updated to {} from {} at {} on {}".format(self.event_dict["Context: Data idString"], self.event_dict["Context: Data Value"], self.event_dict["Revision: Data Value"], self.event_dict["Context: Amend Members"], self.event_dict["Context: Amend Time"], self.event_dict["Context: Amend Date"])
        elif(self.amend_action=="Add"):
            return "For {}, {} is or requested to be added from {} at {} on {}".format(self.event_dict["Context: Data idString"], (self.event_dict["Revision: Data Value"] if self.event_dict["Revision: Data Value"] is not None else self.event_dict["Context: Data Value"] if self.event_dict["Context: Data Value"] is not None else self.event_dict["Revision: Data Value"]), self.event_dict["Context: Amend Members"], self.event_dict["Context: Amend Time"], self.event_dict["Context: Amend Date"])
        elif(self.amend_action=="Delete"):
            return "For {}, {} is or requested to be removed from {} at {} on {}".format(self.event_dict["Context: Data idString"], self.event_dict["Revision: Data Value"] if self.event_dict["Revision: Data Value"] is not None else self.event_dict["Context: Data Value"] if self.event_dict["Context: Data Value"] is not None else self.event_dict["Revision: Data Value"], self.event_dict["Context: Amend Members"], self.event_dict["Context: Amend Time"], self.event_dict["Context: Amend Date"])
        else:
            raise AttributeError('Deliver Action {} is not supported for Deliver Meeting Data event'.format(self.amend_action))



class amend_meeting_data(event_class):
    def __init__(self, eventDict, amend_action, trigger):
        super().__init__("Amend Meeting Data", {"Context: Meeting Members": None, "Revision: Meeting Members": None, "Context: Meeting Agenda": None, "Revision: Meeting Agenda": None, "Context: Meeting Name": None, "Context: Meeting Location": None, "Revision: Meeting Location": None, "Context: Meeting Date": None, "Revision: Meeting Date": None, "Context: Meeting Time": None, "Revision: Meeting Time": None, "Context: Amend Date": None, "Revision: Amend Date": None, "Context: Amend Time": None, "Revision: Amend Time": None, "Context: Amend Members": None,"Revision: Amend Members": None}, eventDict, trigger)
        self.prep_args()
        self.amend_action = amend_action if amend_action.strip()!="" else "Update"
        self.masked_templates = [
                                    ("For {} among {} at {} on {} at {} to discuss {}, meeting members is or requested to be updated to {} from {}" ,"Update"), 
                                    ("For {} among {} at {} on {} at {} to discuss {}, date is or requested to be updated to {} from {}", "Update"), 
                                    ("For {} among {} at {} on {} at {} to discuss {}, time is or requested to be updated to {} from {}", "Update"), 
                                    ("For {} among {} at {} on {} at {} to discuss {}, location is or requested to be updated to {} from {}", "Update"), 
                                    ("For {} among {} at {} on {} at {} to discuss {}, agenda is or requested to be updated to {} from {}", "Update"), 
                                    ("For {} among {} at {} on {} at {} to discuss {}, meeting members {} is or requested to be added from {}", "Add"), 
                                    ("For {} among {} at {} at {} to discuss {}, date {} is or requested to be added from {}", "Add"), 
                                    ("For {} among {} on {} at {} to discuss {}, time {} is or requested to be added from {}", "Add"), 
                                    ("For {} among {} at {} on {} to discuss {}, location {} is or requested to be added from {}", "Add"), 
                                    ("For {} among {} at {} on {} at {}, agenda {} is or requested to be added from {}", "Add"), 
                                    ("For {} among {} at {} on {} at {} to discuss {}, meeting members {} is or requested to be removed from {}", "Remove")
                                ]
    def fill_template(self):
        if(event_class.stats_dict.get("Amend_Meeting_Data: " + self.amend_action) is None):
             event_class.stats_dict["Amend_Meeting_Data: " + self.amend_action] = 0
        event_class.stats_dict["Amend_Meeting_Data: " + self.amend_action] += 1
        revisions_requested = [self.event_dict[x].replace("|", "").strip() for x in self.event_dict.keys() if x.find("Revision:")>=0 and x!=self.event_dict[x].replace("|", "").strip()]
        if(self.amend_action=="Update" and (self.event_dict.get("Revision: Meeting Members") is not None and self.event_dict["Revision: Meeting Members"].replace("|", "").strip()!="Revision: Meeting Members")):
            return 'For {} among {} at {} on {} at {} to discuss {}, meeting members is or requested to be updated to {} from {}'.format(self.event_dict["Context: Meeting Name"] ,self.event_dict["Context: Meeting Members"], self.event_dict["Context: Meeting Time"], self.event_dict["Context: Meeting Date"], self.event_dict["Context: Meeting Location"], self.event_dict["Context: Meeting Agenda"], self.event_dict["Revision: Meeting Members"], self.event_dict["Context: Amend Members"])
        elif((len(revisions_requested)<=0) or (self.amend_action=="Update" and (self.event_dict.get("Revision: Meeting Date") is not None and self.event_dict["Revision: Meeting Date"].replace("|", "").strip()!="Revision: Meeting Date"))):
            return "For {} among {} at {} on {} at {} to discuss {}, date is or requested to be updated to {} from {}".format(self.event_dict["Context: Meeting Name"] ,self.event_dict["Context: Meeting Members"], self.event_dict["Context: Meeting Time"], self.event_dict["Context: Meeting Date"], self.event_dict["Context: Meeting Location"], self.event_dict["Context: Meeting Agenda"], self.event_dict["Revision: Meeting Date"], self.event_dict["Context: Amend Members"])
        elif(self.amend_action=="Update" and (self.event_dict.get("Revision: Meeting Time") is not None and self.event_dict["Revision: Meeting Time"].replace("|", "").strip()!="Revision: Meeting Time")):
            return "For {} among {} at {} on {} at {} to discuss {}, time is or requested to be updated to {} from {}".format(self.event_dict["Context: Meeting Name"] ,self.event_dict["Context: Meeting Members"], self.event_dict["Context: Meeting Time"], self.event_dict["Context: Meeting Date"], self.event_dict["Context: Meeting Location"], self.event_dict["Context: Meeting Agenda"], self.event_dict["Revision: Meeting Time"], self.event_dict["Context: Amend Members"])
        elif(self.amend_action=="Update" and (self.event_dict.get("Revision: Meeting Location") is not None and self.event_dict["Revision: Meeting Location"].replace("|", "").strip()!="Revision: Meeting Location")):
            return "For {} among {} at {} on {} at {} to discuss {}, location is or requested to be updated to {} from {}".format(self.event_dict["Context: Meeting Name"] ,self.event_dict["Context: Meeting Members"], self.event_dict["Context: Meeting Time"], self.event_dict["Context: Meeting Date"], self.event_dict["Context: Meeting Location"], self.event_dict["Context: Meeting Agenda"], self.event_dict["Revision: Meeting Location"], self.event_dict["Context: Amend Members"])
        elif(self.amend_action=="Update" and (self.event_dict.get("Revision: Meeting Agenda") is not None and self.event_dict["Revision: Meeting Agenda"].replace("|", "").strip()!="Revision: Meeting Agenda")):
            return "For {} among {} at {} on {} at {} to discuss {}, agenda is or requested to be updated to {} from {}".format(self.event_dict["Context: Meeting Name"] ,self.event_dict["Context: Meeting Members"], self.event_dict["Context: Meeting Time"], self.event_dict["Context: Meeting Date"], self.event_dict["Context: Meeting Location"], self.event_dict["Context: Meeting Agenda"], self.event_dict["Revision: Meeting Agenda"], self.event_dict["Context: Amend Members"])
        elif(self.amend_action=="Add" and (self.event_dict.get("Revision: Meeting Members") is not None and self.event_dict["Revision: Meeting Members"].replace("|", "").strip()!="Revision: Meeting Members")):
            return "For {} among {} at {} on {} at {} to discuss {}, meeting members {} is or requested to be added from {}".format(self.event_dict["Context: Meeting Name"] ,self.event_dict["Context: Meeting Members"], self.event_dict["Context: Meeting Time"], self.event_dict["Context: Meeting Date"], self.event_dict["Context: Meeting Location"], self.event_dict["Context: Meeting Agenda"], self.event_dict["Revision: Meeting Members"], self.event_dict["Context: Amend Members"])
        elif(self.amend_action=="Add" and (self.event_dict.get("Revision: Meeting Date") is not None and self.event_dict["Revision: Meeting Date"].replace("|", "").strip()!="Revision: Meeting Date")):
            return "For {} among {} at {} at {} to discuss {}, date {} is or requested to be added from {}".format(self.event_dict["Context: Meeting Name"] ,self.event_dict["Context: Meeting Members"], self.event_dict["Context: Meeting Time"], self.event_dict["Context: Meeting Location"], self.event_dict["Context: Meeting Agenda"], self.event_dict["Revision: Meeting Date"], self.event_dict["Context: Amend Members"])
        elif(self.amend_action=="Add" and (self.event_dict.get("Revision: Meeting Time") is not None and self.event_dict["Revision: Meeting Time"].replace("|", "").strip()!="Revision: Meeting Time")):
            return "For {} among {} on {} at {} to discuss {}, time {} is or requested to be added from {}".format(self.event_dict["Context: Meeting Name"] ,self.event_dict["Context: Meeting Members"], self.event_dict["Context: Meeting Date"], self.event_dict["Context: Meeting Location"], self.event_dict["Context: Meeting Agenda"], self.event_dict["Revision: Meeting Time"], self.event_dict["Context: Amend Members"])
        elif(self.amend_action=="Add" and (self.event_dict.get("Revision: Meeting Location") is not None and self.event_dict["Revision: Meeting Location"].replace("|", "").strip()!="Revision: Meeting Location")):
            return "For {} among {} at {} on {} to discuss {}, location {} is or requested to be added from {}".format(self.event_dict["Context: Meeting Name"] ,self.event_dict["Context: Meeting Members"], self.event_dict["Context: Meeting Time"], self.event_dict["Context: Meeting Date"], self.event_dict["Context: Meeting Agenda"], self.event_dict["Revision: Meeting Location"], self.event_dict["Context: Amend Members"])
        elif(self.amend_action=="Add" and (self.event_dict.get("Revision: Meeting Agenda") is not None and self.event_dict["Revision: Meeting Agenda"].replace("|", "").strip()!="Revision: Meeting Agenda")):
            return "For {} among {} at {} on {} at {}, agenda {} is or requested to be added from {}".format(self.event_dict["Context: Meeting Name"] ,self.event_dict["Context: Meeting Members"], self.event_dict["Context: Meeting Time"], self.event_dict["Context: Meeting Date"], self.event_dict["Context: Meeting Location"], self.event_dict["Revision: Meeting Agenda"], self.event_dict["Context: Amend Members"])
        elif(self.amend_action=="Remove" and (self.event_dict.get("Revision: Meeting Members") is not None and self.event_dict["Revision: Meeting Members"].replace("|", "").strip()!="Revision: Meeting Members")):
            return "For {} among {} at {} on {} at {} to discuss {}, meeting members {} is or requested to be removed from {}".format(self.event_dict["Context: Meeting Name"] ,self.event_dict["Context: Meeting Members"], self.event_dict["Context: Meeting Time"], self.event_dict["Context: Meeting Date"], self.event_dict["Context: Meeting Location"], self.event_dict["Context: Meeting Agenda"], self.event_dict["Revision: Meeting Members"], self.event_dict["Context: Amend Members"])
        else:
            raise AttributeError('Amend {} is not supported for Amend Meeting Data event and/or \n the request dictionaries are probably empty {}'.format(self.amend_action, json.dumps(self.event_dict, indent = 4)))


def extract_args_from_labels(label, sentence, event_type):
    arg_dict = {}
    idx = 0
    arg_name, arg_value = "", ""
    while(True):
        if(idx>=len(label)):
            break
        if(label[idx]!='O'):
            lbl = ' '.join(label[idx].rsplit(":")[1:]).replace("  ", ": ")#.replace('B-', '').replace('I-', '')
        else: lbl = label[idx]
        if(lbl.find("B-")>=0):
            if(arg_name!=""):
                if(arg_dict.get(arg_name) is None):
                    arg_dict[arg_name] = []
                arg_dict[arg_name].append(arg_value)
                arg_name, arg_value = "", ""
            arg_name = lbl.replace("B-", "").replace('I-', '').replace("_", " ")
            arg_value = sentence[idx]
        elif(lbl.find("I-")>=0):
            arg_value += " " + sentence[idx]
        else:
            if(arg_name!=""):
                if(arg_dict.get(arg_name) is None):
                    arg_dict[arg_name] = []
                arg_dict[arg_name].append(arg_value)
                arg_name, arg_value = "", ""
        idx += 1
    if(arg_name!=""):
        if(arg_dict.get(arg_name) is None):
                arg_dict[arg_name] = []
        arg_dict[arg_name].append(arg_value)
    arg_dict["event_type"] = event_type.replace("_", " ")
    for arg_name in arg_dict:
        arg_value = arg_dict[arg_name]
        if(type(arg_value)==type([])):
            if(arg_name.lower().find('member')>=0):
                join_delim = " and "
            else:
                join_delim = " "
            arg_value = join_delim.join(arg_value)
            arg_value = re.sub('[ \t]+', ' ', arg_value)
            arg_dict[arg_name] = arg_value
    import pprint
    return arg_dict




template_function_call = {
"Deliver Action Data": deliver_action_data,"Request Action": request_action, 'Deliver Meeting Data':deliver_meeting_data, 'Request Meeting': request_meeting,
"Request Data": request_data, "Deliver Data": deliver_data, "Request Meeting Data": request_meeting_data,
"Request Action Data": request_action_data,
"Amend Data": amend_data,
"Amend Meeting Data": amend_meeting_data,
}


from glob import glob
import json, traceback

def gen_template(turn_event_trigger, turn_events, sentence, events, turn_event_extra):
    while(type(turn_event_trigger)!=type({})):
        turn_event_trigger = ast.literal_eval(turn_event_trigger)
    extracted_args = extract_args_from_labels(turn_events, sentence, events)
    try:
        if(events in ["Deliver_Action_Data", "Deliver_Meeting_Data"]):
            extracted_args.pop("Deliver Members")    
    except:
        pass
    try:
        extracted_args["Deliver Members"] = extracted_args.pop("Deliver members")
    except:
        pass
    try:
        if(events != "Request_Action"):
            extracted_args.pop("Request Members")    
    except:
        pass
    try:
        extracted_args.pop("Data Owner")    
    except:
        pass
    try:
        extracted_args["Action Members"] = extracted_args.pop("Request Members")
    except:
        pass
    template_dict_copy = copy.deepcopy(template_dict[events.replace("_", " ")])
    for args in extracted_args:
        template_dict_copy[args] = extracted_args[args]
    template_dict_copy["trigger"] = turn_event_trigger["words"]
    extra_type, extra_val = turn_event_extra.split(' : ') if turn_event_extra != "" else ("", "")
    template_dict_copy.pop('event_type')
    try:
        template = template_function_call[extracted_args["event_type"]](template_dict_copy, extra_val, template_dict_copy["trigger"])
    except:
        print(sentence)
        traceback.print_exc()
        #sys.exit(-1)
        _ = input("What to do?")
    template_final = template.get_filled_template()
    return re.sub('[ \t]+', ' ', template_final)


def gen_templates_all():
    all_files = glob('./data/full_data/*.json')
    todo, removed_owner, removed_deliver_members, removed_request_members = 0, 0, 0, 0
    for file in all_files:
        f = open(file)
        jsn = json.load(f)
        for turns, sentence in zip(jsn["events"], jsn["sentences"]):
            for events in jsn["events"][turns]:
                if(events in ["O", "Amend_Action_Data"]): 
                    if(events!="O"):
                        todo += 1
                    continue
                turn_event_labels = jsn["events"][turns][events]["labels"]#list of labels for each event of same type
                turn_event_triggers=jsn["events"][turns][events]["triggers"]#for each event of same type --- the triggers
                turn_event_extras = jsn["events"][turns][events]["extras"]#for each event of same type --- meta roles
                for turn_events, turn_event_extra, turn_event_trigger in zip(turn_event_labels, turn_event_extras, turn_event_triggers):
                    while(type(turn_event_trigger)!=type({})):
                        turn_event_trigger = ast.literal_eval(turn_event_trigger)
                    extracted_args = extract_args_from_labels(turn_events, sentence, events)
                    try:
                        removed_deliver_members += 1
                        if(events in ["Deliver_Action_Data", "Deliver_Meeting_Data"]):
                            extracted_args.pop("Deliver Members")    
                    except:
                        pass
                    try:
                        extracted_args["Deliver Members"] = extracted_args.pop("Deliver members")
                    except:
                        pass
                    try:
                        removed_request_members += 1
                        if(events != "Request_Action"):
                            extracted_args.pop("Request Members")    
                    except:
                        pass
                    try:
                        removed_owner +=1
                        extracted_args.pop("Data Owner")    
                    except:
                        pass
                    try:
                        extracted_args["Action Members"] = extracted_args.pop("Request Members")
                    except:
                        pass
                    template_dict_copy = copy.deepcopy(template_dict[events.replace("_", " ")])
                    for args in extracted_args:
                        template_dict_copy[args] = extracted_args[args]
                    template_dict_copy["trigger"] = turn_event_trigger["words"]
                    extra_type, extra_val = turn_event_extra.split(' : ') if turn_event_extra != "" else ("", "")
                    template_dict_copy.pop('event_type')
                    try:
                        template = template_function_call[extracted_args["event_type"]](template_dict_copy, extra_val, template_dict_copy["trigger"])
                        print('>>>', template.get_filled_template())
                    except Exception as e:
                        print(sentence, file)
                        traceback.print_exc()
                        #sys.exit(-1)
                        _ = input("What to do?")
                
from pprint import pprint
if __name__ == '__main__':
    gen_templates_all()
    pprint(event_class.stats_dict)