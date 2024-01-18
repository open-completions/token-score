import datetime


class Event:
    def __init__(self, name, event_date, event_time):
        self.name = name
        self.event_date = event_date
        self.event_time = event_time

    def __str__(self):
        return f"Event: {self.name} at {self.event_date} {self.event_time}"


class Scheduler:
    def __init__(self):
        self.events = []

    def add_event(self, event):
        self.events.append(event)

    def remove_event(self, event_name):
        self.events = [event for event in self.events if event.name != event_name]

    def check_events(self):
        current_date = datetime.datetime.now().date()
        current_time = datetime.datetime.now().time()
        due_events = [
            event
            for event in self.events
            if event.event_date == current_date and event.event_time <= current_time
        ]
        return due_events


def main():
    scheduler = Scheduler()
    scheduler.add_event(
        Event("Meeting", datetime.date(2024, 1, 17), datetime.time(10, 30))
    )
    scheduler.add_event(
        Event("Doctor Appointment", datetime.date(2024, 1, 17), datetime.time(12, 0))
    )

    due_events = scheduler.check_events()
    if due_events:
        print("Due Events:")
        for event in due_events:
            print(event)
    else:
        print("No events due at this time.")


if __name__ == "__main__":
    main()
