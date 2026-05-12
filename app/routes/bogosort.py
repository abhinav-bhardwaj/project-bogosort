from flask import Blueprint, redirect, render_template, url_for, request, session
import threading
import time
import uuid
import logging
from app.services.sorting_service import SortingService
from app.services.session_manager import SessionManager

logger = logging.getLogger(__name__)

bogosort_demo = Blueprint('bogosort', __name__, url_prefix='/sort-demo')

session_manager = SessionManager(timeout_minutes=30)
sorting_threads = {}


def get_session_id():
    """Get or create session ID."""
    if 'bogosort_session_id' not in session:
        session['bogosort_session_id'] = str(uuid.uuid4())
        logger.debug(f"Created sorting session: {session['bogosort_session_id']}")
    return session['bogosort_session_id']


def get_sorting_session():
    """Get the sorting session for the current user."""
    session_id = get_session_id()
    return session_manager.get_or_create_session(session_id)


def reset_session():
    """Reset the sorting session."""
    session_id = get_session_id()
    if session_id in session_manager.sessions:
        del session_manager.sessions[session_id]
    if session_id in sorting_threads:
        del sorting_threads[session_id]
    return session_manager.get_or_create_session(session_id)


@bogosort_demo.route('/', methods=['GET', 'POST'])
def bogosort():
    """Main bogosort/mergesort route."""
    if request.method == 'POST':
        return handle_post()
    else:
        return handle_get()


def handle_post():
    """Handle POST request to start sorting."""
    algorithm = request.form.get('algorithm', 'bogosort')
    seed_str = request.form.get('seed', '').strip()
    try:
        seed = int(seed_str) if seed_str else None
    except ValueError:
        logger.warning(f"Invalid seed value: {seed_str}, using random seed")
        seed = None

    sorting_session = get_sorting_session()
    session_id = get_session_id()

    # Only start if not already running
    if sorting_session['state'] != 'running':
        try:
            # Load words and save distribution
            words, counts = SortingService.load_shuffled_toxic_words(seed=seed)
            SortingService.save_distribution_plot(words, counts, 'app/static/word_distribution.png')

            # Mark as running and start thread
            sorting_session['state'] = 'running'
            sorting_session['algorithm'] = algorithm
            sorting_session['seed'] = seed_str

            if algorithm == 'mergesort':
                thread = threading.Thread(
                    target=background_mergesort,
                    args=(words, counts, 'app/static/mergesort_sorting.gif', sorting_session, seed, session_id),
                    daemon=True
                )
            else:
                thread = threading.Thread(
                    target=background_bogosort,
                    args=(words, counts, 'app/static/bogosort_sorting.gif', sorting_session, seed, session_id),
                    daemon=True
                )

            sorting_threads[session_id] = thread
            thread.start()

        except Exception as e:
            sorting_session['state'] = 'error'
            sorting_session['error'] = str(e)

    # Always redirect to GET to show the current state
    return redirect(url_for('bogosort.bogosort'))


def handle_get():
    """Handle GET request to display current state."""
    sorting_session = get_sorting_session()
    state = sorting_session.get('state')
    algorithm = sorting_session.get('algorithm', 'bogosort')
    seed = sorting_session.get('seed', '')

    # Load static file URLs
    dist_url = url_for('static', filename='word_distribution.png') + f'?v={int(time.time())}'
    if algorithm == 'mergesort':
        gif_url = url_for('static', filename='mergesort_sorting.gif') + f'?v={int(time.time())}'
    else:
        gif_url = url_for('static', filename='bogosort_sorting.gif') + f'?v={int(time.time())}'

    # Determine what to show based on state
    if state == 'running':
        # Sorting is in progress - show spinner only
        return render_template(
            'sort-demo.html',
            dist_url=dist_url,
            gif_url=gif_url,
            show_form=False,
            show_spinner=True,
            show_gif=False,
            algorithm=algorithm,
            seed=seed
        )

    elif state == 'done':
        # Sorting completed - show GIF and results
        return render_template(
            'sort-demo.html',
            dist_url=dist_url,
            gif_url=gif_url,
            show_form=False,
            show_spinner=False,
            show_gif=True,
            sorted=sorting_session.get('sorted', False),
            iteration=sorting_session.get('final_iteration', 0),
            algorithm=algorithm,
            seed=seed
        )

    elif state == 'error':
        # Error occurred - show form with error message
        return render_template(
            'sort-demo.html',
            dist_url=dist_url,
            gif_url=gif_url,
            show_form=True,
            show_spinner=False,
            show_gif=False,
            error=sorting_session.get('error', 'Unknown error'),
            algorithm='bogosort',
            seed=''
        )

    else:
        # Initial state (None) - show distribution and form
        try:
            words, counts = SortingService.load_shuffled_toxic_words(seed=None)
            SortingService.save_distribution_plot(words, counts, 'app/static/word_distribution.png')
        except Exception as e:
            return render_template(
                'sort-demo.html',
                dist_url=dist_url,
                gif_url=gif_url,
                show_form=True,
                show_spinner=False,
                show_gif=False,
                error=f'Failed to load data: {str(e)}',
                algorithm='bogosort',
                seed=''
            )

        return render_template(
            'sort-demo.html',
            dist_url=dist_url,
            gif_url=gif_url,
            show_form=True,
            show_spinner=False,
            show_gif=False,
            algorithm='bogosort',
            seed=''
        )


@bogosort_demo.route('/stop', methods=['POST'])
def stop_sorting():
    """Stop the current sorting operation."""
    sorting_session = get_sorting_session()
    sorting_session['stop_flag'] = True
    return redirect(url_for('bogosort.bogosort'))


@bogosort_demo.route('/reset', methods=['GET'])
def reset():
    """Reset to initial state."""
    reset_session()
    return redirect(url_for('bogosort.bogosort'))


def background_bogosort(words, counts, gif_filename, sorting_session, seed=None, session_id=None):
    """Run bogosort in background."""
    try:
        stop_flag = {'stop': False}
        snapshots = SortingService.bogosort_snapshots(
            words, counts, max_iterations=1000, seed=seed, stop_flag=stop_flag
        )
        SortingService.save_sort_animation(snapshots, gif_filename, title='Bogosort Animation')

        sorting_session['state'] = 'done'
        sorting_session['final_iteration'] = snapshots[-1][1]
        sorting_session['sorted'] = SortingService.is_sorted([x[1] for x in snapshots[-1][0]])
    except Exception as e:
        sorting_session['state'] = 'error'
        sorting_session['error'] = str(e)
    finally:
        if session_id in sorting_threads:
            del sorting_threads[session_id]


def background_mergesort(words, counts, gif_filename, sorting_session, seed=None, session_id=None):
    """Run mergesort in background."""
    try:
        stop_flag = {'stop': False}
        snapshots = SortingService.mergesort_snapshots(
            words, counts, seed=seed, stop_flag=stop_flag
        )
        SortingService.save_sort_animation(snapshots, gif_filename, title='MergeSort Animation')

        sorting_session['state'] = 'done'
        sorting_session['final_iteration'] = snapshots[-1][1]
        sorting_session['sorted'] = SortingService.is_sorted([x[1] for x in snapshots[-1][0]])
    except Exception as e:
        sorting_session['state'] = 'error'
        sorting_session['error'] = str(e)
    finally:
        if session_id in sorting_threads:
            del sorting_threads[session_id]
