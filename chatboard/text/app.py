from chatboard.text.app_manager import app_manager




def add_chatboard(app):
    
    @app.get('/chatboard/metadata')
    def get_chatboard_metadata():
        app_metadata = app_manager.get_metadata()
        return {"metadata": app_metadata}
    

    print("Chatboard added to app.")
