import React from 'react';
import '../styles/dashboard.css';

const SearchBox = ({ searchTerm, setSearchTerm, setActivePage }) => {
    return (
        <div className="top-bar">
            <div className="search-box">
                <i className="fa fa-search search-icon" />
                <input
                    type="text"
                    placeholder="Search"
                    value={searchTerm}
                    onChange={e => {
                        setSearchTerm(e.target.value);
                        if (setActivePage) setActivePage(1);
                    }}
                />
                <button>Search</button>
            </div>
            <div className="icons">
                <i className="fa fa-bell" />
                <i className="fa fa-question-circle" />
            </div>
        </div>
    );
};

export default SearchBox;
