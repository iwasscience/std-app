import os
import flask
from flask import Flask
import dash
import dash_html_components as html
#WAS IST MIT DASH CORE COMPONENTS? (noch benutz ich sie ja nicht)
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from flask import render_template,url_for,flash,redirect,request,Blueprint

login_manager = LoginManager()

server = flask.Flask(__name__)

server.config['SECRET_KEY'] = 'mysecretkey'
server.config['SECRET_KEY'] = 'mysecretkey'
basedir = os.path.abspath(os.path.dirname(__file__))
server.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'data.sqlite')
server.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(server)
Migrate(server,db)

login_manager.init_app(server)
login_manager.login_view = 'login'


app = dash.Dash(
    __name__,
    server=server,
    routes_pathname_prefix='/dash/'
    )
