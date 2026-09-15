def get_color(column_percent, reverse=False):
    if column_percent == None:
        color = 'hsl(360, 100%, 100%)'
    else:
        if column_percent > 0:
            if column_percent <= 3:
                color = 'hsl(96, 85%, 85%)'
            elif column_percent <= 5:
                color = 'hsl(96, 85%, 80%)'
            elif column_percent <= 10:
                color = 'hsl(96, 85%, 75%)'
            elif column_percent <= 20:
                color = 'hsl(96, 85%, 70%)'
            elif column_percent <= 30:
                color = 'hsl(96, 85%, 65%)'
            elif column_percent > 30:
                color = 'hsl(96, 85%, 60%)'
        elif column_percent < 0:
            if column_percent >= -3:
                color = 'hsl(0, 85%, 85%)'
            elif column_percent >= -5:
                color = 'hsl(0, 85%, 80%)'
            elif column_percent >= -10:
                color = 'hsl(0, 85%, 75%)'
            elif column_percent >= -20:
                color = 'hsl(0, 85%, 70%)'
            elif column_percent >= -30:
                color = 'hsl(0, 85%, 65%)'
            elif column_percent < -30:
                color = 'hsl(0, 85%, 60%)'
        else:
                color = 'hsl(360, 100%, 100%)'
    return color

def get_font_color(column_percent, leadership=False):
    if leadership == False:
        if column_percent == None:
            color = 'black'
        else:
            if column_percent > 0:
                color = 'hsl(120, 60%, 30%)'
            elif column_percent < 0:
                color = 'hsl(0, 60%, 30%)'
            else:
                color = 'black'
        return color
    else:
        if column_percent == None:
            color = 'black'
        else:
            if column_percent > 0:
                color = '#66c296'
            elif column_percent < 0:
                color = '#da6c53'
            else:
                color = 'black'
        return color

def get_reverse_font_color(column_percent):
        if column_percent == None:
            color = 'black'
        else:
            if column_percent > 0:
                color = 'hsl(0, 60%, 30%)'
            elif column_percent < 0:
                color = 'hsl(120, 60%, 30%)'
            else:
                color = 'black'
        return color

def get_reverse_color(column_percent):
    if column_percent == None:
        color = 'hsl(360, 100%, 100%)'
    else:
        if column_percent > 0:
            if column_percent <= 3:
                color = 'hsl(0, 85%, 85%)'
            elif column_percent <= 5:
                color = 'hsl(0, 85%, 80%)'
            elif column_percent <= 10:
                color = 'hsl(0, 85%, 75%)'
            elif column_percent <= 20:
                color = 'hsl(0, 85%, 70%)'
            elif column_percent <= 30:
                color = 'hsl(0, 85%, 65%)'
            elif column_percent > 30:
                color = 'hsl(0, 85%, 60%)'
        elif column_percent < 0:
            if column_percent >= -3:
                color = 'hsl(96, 85%, 85%)'
            elif column_percent >= -5:
                color = 'hsl(96, 85%, 80%)'
            elif column_percent >= -10:
                color = 'hsl(96, 85%, 75%)'
            elif column_percent >= -20:
                color = 'hsl(96, 85%, 70%)'
            elif column_percent >= -30:
                color = 'hsl(96, 85%, 65%)'
            elif column_percent < -30:
                color = 'hsl(96, 85%, 60%)'
        else:
                color = 'hsl(360, 100%, 100%)'
    return color


